import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    # # Uncomment this code if you want to run the notebook on marimo cloud
    # import micropip  # type: ignore

    # await micropip.install("Mastodon.py")
    # await micropip.install("loguru")
    return


@app.cell
def _():
    import marimo as mo
    import pickle
    import time
    import altair as alt
    from sklearn.manifold import TSNE
    import pandas as pd
    import numpy as np

    from byota.embeddings import (
        EmbeddingService,
        LLamafileEmbeddingService,
        OllamaEmbeddingService,
    )
    import byota.config as config
    import byota.layout as layout
    import byota.mastodon
    from byota.search import SearchService

    return (
        EmbeddingService,
        LLamafileEmbeddingService,
        OllamaEmbeddingService,
        SearchService,
        TSNE,
        alt,
        byota,
        config,
        layout,
        mo,
        np,
        pd,
        pickle,
        time,
    )


@app.cell
def _(byota, config, mo):
    # internal variables

    # credentials filename
    cred_filename = "auth.json"

    # dump files for offline mode
    dataframes_data_file = "dump_dataframes.pkl"
    embeddings_data_file = "dump_embeddings.pkl"

    accounts = config.load_accounts(cred_filename)

    clients = {}
    for account in accounts:
        clients[account.name] = byota.mastodon.login(
            access_token=account.MASTODON_ACCESS_TOKEN,
            api_base_url=account.MASTODON_API_BASE_URL,
        )

    mo.stop(
        len(clients) < len(accounts),
        mo.md("**Authentication error: please check your credentials.**").center(),
    )
    return accounts, clients, dataframes_data_file, embeddings_data_file


@app.cell
def _(accounts, layout):
    configuration_form = layout.create_configuration_form(accounts)
    configuration_form
    return (configuration_form,)


@app.cell
def _(
    LLamafileEmbeddingService,
    OllamaEmbeddingService,
    byota,
    configuration_form,
    layout,
    mo,
):
    # check for anything invalid in the form
    mo.stop(
        layout.invalid_form(configuration_form),
        mo.md("**Submit the form to continue.**").center(),
    )

    # instantiate an embedding service (and break if it does not work)
    if configuration_form.value["emb_server"] == "llamafile":
        embedding_service = LLamafileEmbeddingService(
            configuration_form.value["emb_server_url"]
        )
    else:
        embedding_service = OllamaEmbeddingService(
            configuration_form.value["emb_server_url"],
            configuration_form.value["emb_server_model"],
        )

    mo.stop(
        not embedding_service.is_working(),
        mo.md(
            f"**Cannot access {configuration_form.value['emb_server']} embedding server.**"
        ),
    )

    # collect the names of the timelines we want to download from
    timelines = byota.mastodon.extract_selected_timelines(configuration_form.value)

    # set offline mode
    offline_mode = configuration_form.value["offline_mode"]

    # choose what to read from cache
    cached_dataframes = offline_mode
    cached_embeddings = offline_mode
    return cached_dataframes, cached_embeddings, embedding_service, timelines


@app.cell
def _(mo):
    mo.md("### Downloading timelines").center()
    return


@app.cell
def _(byota, cached_dataframes, clients, dataframes_data_file, mo, timelines):
    dataframes = byota.mastodon.download_to_dataframes(
        clients, timelines, cached_dataframes, dataframes_data_file
    )

    mo.stop(dataframes is None, mo.md("**Issues building dataframes**"))
    return (dataframes,)


@app.cell
def _(dataframes, mo):
    mo.md(f"""
    ### Calculating embeddings for the downloaded timeline{"s" if len(dataframes.keys())>1 else ""}
    """).center()
    return


@app.cell
def _(
    build_cache_embeddings,
    cached_embeddings,
    dataframes,
    embedding_service,
    embeddings_data_file,
    mo,
):
    # calculate embeddings
    embeddings = build_cache_embeddings(
        embedding_service, dataframes, cached_embeddings, embeddings_data_file
    )
    mo.stop(embeddings is None, mo.md("**Issues calculating embeddings**"))
    return (embeddings,)


@app.cell
def _(TSNE, alt, dataframes, embeddings, mo, np, pd):
    def tsne(dataframes, embeddings, perplexity, random_state=42):
        """Runs dimensionality reduction using TSNE on the input embeddings.
        Returns dataframes containing status id, text, and 2D coordinates
        for plotting.
        """
        tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)

        all_embeddings = np.concatenate([v for v in embeddings.values()])
        all_projections = tsne.fit_transform(all_embeddings)

        dfs = []
        start_idx = 0
        end_idx = 0
        for kk in embeddings:
            end_idx += len(embeddings[kk])
            df = dataframes[kk]
            df["x"] = all_projections[start_idx:end_idx, 0]
            df["y"] = all_projections[start_idx:end_idx, 1]
            df["label"] = kk
            dfs.append(df)
            start_idx = end_idx

        return pd.concat(dfs, ignore_index=True), all_embeddings

    df_, all_embeddings = tsne(dataframes, embeddings, perplexity=16)

    chart = mo.ui.altair_chart(
        alt.Chart(df_, title="Timeline Visualization", height=500, width=880)
        .mark_point()
        .encode(x="x", y="y", color="label")
    )
    return all_embeddings, chart, df_


@app.cell
def _(chart, mo):
    mo.vstack(
        [
            mo.md("# Embeddings visualization").center(),
            mo.md("""
                In this section, you can see posts from different timelines represented as points on a plane:
                You can click on a timeline label on the top right to highlight only posts from that timeline.
                If you select one or more points, you will see them in the table below the plot.
                By clicking on the column names (e.g. `label`, `text`) you can sort them, wrap text (to see full
                post contents), or search their content.
            """),
            chart,
            mo.ui.table(
                chart.value[["id", "label", "text"]],
                column_widths={"text": 800},
                selection=None,
            )
            if len(chart.value) > 0
            else chart.value,
        ]
    )
    return


@app.cell
def _(embeddings, layout, mo):
    mo.stop(embeddings is None)

    query_form = layout.create_query_form()
    mo.vstack([mo.md("# Timeline search"), query_form])
    return (query_form,)


@app.cell
def _(SearchService, all_embeddings, df_, embedding_service, mo, query_form):
    search_service = SearchService(all_embeddings, embedding_service)
    indices = search_service.most_similar_indices(query_form.value)
    mo.ui.table(
        df_.iloc[indices][["label", "text"]],
        column_widths={"text": 940},
        selection=None,
    )
    return


@app.cell
def _(accounts, dataframes, embeddings, layout, mo):
    mo.stop(embeddings is None)
    rerank_form = layout.create_rerank_form(
        list(dataframes.keys()), [acc.name for acc in accounts]
    )
    rerank_form
    return (rerank_form,)


@app.cell
def _(
    byota,
    clients,
    dataframes,
    embedding_service,
    embeddings,
    mo,
    np,
    pd,
    rerank_form,
    time,
):
    mo.stop(embeddings is None)

    # check for anything invalid in the form
    mo.stop(rerank_form.value is None, mo.md("**Submit the form to continue.**"))

    timeline_to_rerank = rerank_form.value["timeline_to_rerank"]
    user_account = rerank_form.value["user_account"]

    client_for_user_posts = clients[user_account]

    # download and calculate embeddings of user statuses
    user_statuses = byota.mastodon.get_paginated_statuses(
        client_for_user_posts,
        max_pages=rerank_form.value["num_user_status_pages"],
        exclude_reblogs=rerank_form.value["exclude_reblogs"],
    )
    user_statuses_df = pd.DataFrame(
        byota.mastodon.get_compact_data(user_statuses), columns=["id", "text"]
    )

    # Without any of the user's own posts we cannot build a "goal vector"
    # to re-rank the timeline against, so stop with a helpful message
    # instead of failing later on an empty matrix multiplication.
    mo.stop(
        len(user_statuses_df) == 0,
        mo.md(
            f"""**No posts found for `{user_account}`, so the timeline cannot be re-ranked.**

This can happen when:

- the account hasn't published any posts yet (or only replies/reblogs, which may be excluded above);
- the account's profile is set to **not** be discoverable in search engines (`discoverable = False`), in which case its posts are skipped on purpose;
- the credentials used point to a different account than expected.

Try publishing a few posts, enabling profile discoverability in your Mastodon
settings, or lowering the `exclude_replies`/`exclude_reblogs` options before
submitting the form again."""
        ),
    )

    user_statuses_embeddings = embedding_service.calculate_embeddings(
        user_statuses_df["text"]
    )

    # build an index of most similar statuses to the ones
    # published / boosted by the user
    rerank_start_time = time.time()
    # index is in reverse order (from largest to smallest similarity)
    idx = np.flip(
        # return indices of the sorted list, instead of values
        # we want to get pointers to statuses, not actual similarities
        np.argsort(
            # to measure how much I might like a timeline status,
            # I sum all the similarity values calculated between
            # that status and all the statuses in my feed
            np.sum(
                # dot product is a decent quick'n'dirty way to calculate
                # similarity between two vectors (the more similar they
                # are, the larger the product)
                np.dot(user_statuses_embeddings, embeddings[timeline_to_rerank].T),
                axis=0,
            )
        )
    )

    print(time.time() - rerank_start_time)

    # show everything
    mo.vstack(
        [
            mo.md("""## Your statuses:
    This table shows the content of the posts that are used for re-ranking the timeline. You can change
    their number in the form above (1 page = 20 posts), check them out here, and verify in the table below
    this one how ranking changes depending on the contents you include.
    """),
            mo.ui.table(user_statuses_df, column_widths={"text": 940}, selection=None),
            mo.md("""## Your re-ranked timeline:
    This table shows posts from the synthetic timelines (you can choose between home, local, and public
    in the form above), re-ranked to prioritize the main topics inferred from the posts in the previous table.
    """),
            mo.ui.table(
                # show statuses sorted by idx
                dataframes[timeline_to_rerank].iloc[idx][["label", "text"]],
                column_widths={"text": 940},
                selection=None,
            ),
        ]
    )
    return client_for_user_posts, user_account


@app.cell
def _():
    # # Wanna get some intuition re: the similarity measure?
    # # Here's a simple example: the seven values you get are
    # # the scores for the seven vectors in bbb (the higher
    # # they are, the more similar vectors they have in aaa).
    # # ... Can you tell why the third vector in bbb ([1,1,0,0])
    # # is the most similar to vectors found in aaa?

    # aaa = np.array([
    #           [1,0,0,0],
    #           [0,1,0,0],
    #           [0,0,1,0],
    #           [1,1,0,0],
    #          ]).astype(np.float32)

    # bbb = np.array([
    #           [1,0,0,0],
    #           [0,1,0,0],
    #           [1,1,0,0],
    #           [0,0,1,0],
    #           [0,1,1,0],
    #           [0,0,0,1],
    #           [0,0,1,1],
    #          ]).astype(np.float32)

    # np.sum(np.dot(aaa, bbb.T), axis=0)
    return


@app.cell
def _(layout, mo, rerank_form, user_account):
    mo.stop(rerank_form.value is None)

    tag_form = layout.create_tag_form(user_account)

    mo.vstack(
        [
            mo.md("""
    # Re-Ranking your own posts
    Depending on the timeline you are considering, it might be more or less hard
    to understand how well the re-ranking worked.
    To give you a better sense of the effect of re-ranking, let us take the posts
    you wrote and re-rank them according to some well-known tag.
    Feel free to test the following code with different tags, depending on your
    various interests, and see whether your own posts related to a given interest
    are surfaced by a related tag.
        """),
            tag_form,
        ]
    )
    return (tag_form,)


@app.cell
def _(byota, client_for_user_posts, embedding_service, mo, pd, rerank_form):
    mo.stop(rerank_form.value is None)

    my_posts = byota.mastodon.get_paginated_statuses(
        client_for_user_posts, max_pages=10, exclude_reblogs=True, exclude_replies=True
    )
    my_posts_df = pd.DataFrame(
        byota.mastodon.get_compact_data(my_posts), columns=["id", "text"]
    )

    # Without any of your own posts there is nothing to re-rank, and the
    # downstream similarity computation would fail on an empty matrix.
    mo.stop(
        len(my_posts_df) == 0,
        mo.md(
            """**No posts of yours were found, so there is nothing to re-rank here.**

This can happen when:

- you haven't published any posts yet (replies and reblogs are excluded here);
- your profile is set to **not** be discoverable in search engines
  (`discoverable = False`), in which case your posts are skipped on purpose.

Publish a few posts or enable profile discoverability in your Mastodon
settings, then re-run this notebook."""
        ),
    )

    my_posts_embeddings = embedding_service.calculate_embeddings(my_posts_df["text"])
    return my_posts_df, my_posts_embeddings


@app.cell
def _(
    byota,
    client_for_user_posts,
    embedding_service,
    mo,
    my_posts_df,
    my_posts_embeddings,
    np,
    pd,
    tag_form,
    user_account,
):
    tag_name = f"tag/{tag_form.value}"

    ### TODO: Fix with a client chosen by the user!!!
    tag_posts = byota.mastodon.get_paginated_data(
        client_for_user_posts, tag_name, max_pages=1
    )
    tag_posts_df = pd.DataFrame(
        byota.mastodon.get_compact_data(tag_posts), columns=["id", "text"]
    )
    tag_posts_embeddings = embedding_service.calculate_embeddings(tag_posts_df["text"])

    # calculate the re-ranking index
    my_idx = np.flip(
        np.argsort(np.sum(np.dot(tag_posts_embeddings, my_posts_embeddings.T), axis=0))
    )
    # let us also show the similarity scores used to calculate the index
    my_posts_df["scores"] = np.sum(
        np.dot(tag_posts_embeddings, my_posts_embeddings.T), axis=0
    )

    mo.vstack(
        [
            mo.md(
                f"### Your own posts, re-ranked according to their similarity to posts in {user_account}:{tag_name}"
            ),
            mo.ui.table(
                my_posts_df.iloc[my_idx][["text", "scores"]],
                column_widths={"text": 880},
                selection=None,
            ),
        ]
    )
    # my_posts_df[['text', 'scores']]
    return


@app.cell
def _(EmbeddingService, mo, pickle, time):
    def build_cache_embeddings(
        embedding_service: EmbeddingService,  # type: ignore
        dataframes: dict[str, any],
        cached: bool,
        embeddings_data_file: str,
    ) -> dict[str, any]:
        """Given a dictionary with dataframes from different timelines,
        return another dictionary that contains, for each timeline, the
        respective embeddings calculated with the provided embedding service.
        If cached==True, the `embeddings_data_file` file will be loaded.
        """
        if not cached:
            embeddings = {}
            for k in dataframes:
                with mo.status.progress_bar(
                    total=len(dataframes[k]), title=f"Embedding posts from: {k}"
                ) as bar:
                    print(f"Embedding statuses from timeline: {k}")
                    tt_ = time.time()
                    embeddings[k] = embedding_service.calculate_embeddings(
                        dataframes[k]["text"], bar
                    )
                    print(time.time() - tt_)
            with open(embeddings_data_file, "wb") as f:
                pickle.dump(embeddings, f)
        else:
            print(f"Loading cached embeddings from {embeddings_data_file}")
            try:
                with open(embeddings_data_file, "rb") as f:
                    embeddings = pickle.load(f)
            except FileNotFoundError:
                print(f"File {embeddings_data_file} not found.")
                return None

        return embeddings

    return (build_cache_embeddings,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
