import marimo

__generated_with = "0.13.13"
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

    from byota.embeddings import EmbeddingService, LLamafileEmbeddingService
    import byota.config as config
    import byota.layout as layout
    from byota.search import SearchService

    return (
        EmbeddingService,
        LLamafileEmbeddingService,
        SearchService,
        TSNE,
        alt,
        config,
        layout,
        mo,
        np,
        pd,
        pickle,
        time,
    )


@app.cell
def _():
    # demo dataset files
    dataframes_data_file = "data/dump_dataframes_demo.pkl"
    embeddings_data_file = "data/dump_embeddings_demo.pkl"
    user_statuses_data_file = "data/dump_user_statuses_demo.pkl"
    return dataframes_data_file, embeddings_data_file, user_statuses_data_file


@app.cell
def _(mo):
    mo.md(
        """
    # Build Your Own Timeline Algorithm

    Welcome to BYOTA's demo!

    This small Web application shows some of the things you could do running BYOTA's code on your own timeline.
    As this is open for anyone to use, this version of the code does not connect to any real social network: regardless of what you specify in the Configuration form below, it will load synthetic data (to simulate posts in the home, local, and public timelines) and posts from [my Mastodon account](http://fosstodon.org/@mala).
    Embeddings will be calculated in realtime using a local version of llamafile, serving the `all-minilm` embedding model.

    If you want to use BYOTA with your own data, feel free to check its [⌨️ code](https://github.com/mozilla-ai/byota)
    and [📖 documentation](https://mozilla-ai.github.io/byota/).

    Now, click "submit" in the following Configuration form and... see what happens!
    """
    )
    return


@app.cell
def _(config, layout):
    configuration_form = layout.create_configuration_form(config.mock_account_list())
    configuration_form
    return (configuration_form,)


@app.cell
def _(
    LLamafileEmbeddingService,
    configuration_form,
    dataframes_data_file,
    layout,
    load_dataframes,
    mo,
):
    # check for anything invalid in the form
    mo.stop(
        layout.invalid_form(configuration_form),
        mo.md("**Submit the form to continue.**").center(),
    )

    embedding_service = LLamafileEmbeddingService("http://localhost:8080/embedding")

    mo.stop(
        not embedding_service.is_working(),
        mo.md("**Cannot access embedding server.**"),
    )

    # choose what to read from cache
    cached_embeddings = configuration_form.value["offline_mode"]

    dataframes = load_dataframes(dataframes_data_file)
    mo.stop(dataframes is None, mo.md("**Issues loading dataframes**"))
    return cached_embeddings, dataframes, embedding_service


@app.cell
def _(dataframes, mo):
    mo.stop(dataframes is None)
    mo.md(f"""
    ### Calculating embeddings for the downloaded timeline{"s" if len(dataframes.keys())>1 else ""}.
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

    df_, all_embeddings = tsne(dataframes, embeddings, perplexity=4)

    chart = mo.ui.altair_chart(
        alt.Chart(df_, title="Timeline Visualization", height=500)
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
            chart.value[["id", "label", "text"]]
            if len(chart.value) > 0
            else chart.value,
        ]
    )
    return


@app.cell
def _(embeddings, layout, mo):
    mo.stop(embeddings is None)

    query_form = layout.create_query_form()
    mo.vstack(
        [
            mo.md("# Timeline search"),
            mo.md("""
            Here you can search for the most similar posts to a given one.
            You can either provide a row id (the leftmost column in the previous table) to refer to an existing post,
            or freeform text to look for posts which are similar in content to what you wrote. Some examples:

            - Book suggestions for scifi lovers
            - Digital rights and free software
            - Recipes for vegetarians (warning: sadly you won't get recipes from this dataset!)
            - I like retrocomputing but also bouldering, now what?

        """),
            query_form,
        ]
    )
    return (query_form,)


@app.cell
def _(SearchService, all_embeddings, df_, embedding_service, query_form):
    search_service = SearchService(all_embeddings, embedding_service)
    indices = search_service.most_similar_indices(query_form.value)
    df_.iloc[indices][["label", "text"]]
    return


@app.cell
def _(dataframes, embeddings, layout, mo):
    mo.stop(embeddings is None)

    rerank_form = layout.create_rerank_form(
        list(dataframes.keys()), ["@mala's statuses"]
    )

    mo.vstack(
        [
            mo.md("# Timeline Re-ranking"),
            mo.md("""
        In the previous sections, you saw that embeddings are reasonable descriptors for social media posts,
        as they allow semantic similar statuses to be close in the embedding space. This allows you to use
        the simple concept of *distance between points* to group statuses and search them.

        In this section, you will perform actual timeline re-ranking. To do this, you'll still rely on the
        concept of text similarity, assigning a higher score to those posts which are most similar to *a set
        of other posts*. The set you'll use as a reference is the one of the posts you wrote or
        reposted from others.

        **NOTE**: For the sake of this open demo, the posts are not the ones *you* wrote, but I provided a subset of
        those posted by https://fosstodon.org/@mala (that's me!). This way, you can get a better sense of
        how this would work with some real data rather than a fully synthetic dataset.
        """),
            rerank_form,
        ]
    )
    return (rerank_form,)


@app.cell
def _(
    dataframes,
    embedding_service,
    embeddings,
    load_dataframes,
    mo,
    np,
    rerank_form,
    time,
    user_statuses_data_file,
):
    mo.stop(embeddings is None)

    # check for anything invalid in the form
    mo.stop(rerank_form.value is None, mo.md("**Submit the form to continue.**"))

    timeline_to_rerank = rerank_form.value["timeline_to_rerank"]

    user_statuses_df = load_dataframes(user_statuses_data_file)[
        : 20 * rerank_form.value["num_user_status_pages"]
    ]

    mo.stop(user_statuses_df is None, mo.md("**Issues loading dataframes**"))

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
            user_statuses_df,
            mo.md(f"""## Your re-ranked timeline ({timeline_to_rerank}):
    This table shows posts from the synthetic timelines (you can choose between home, local, and public
    in the form above), re-ranked to prioritize the main topics inferred from the posts in the previous table.
    """),
            # show statuses sorted by idx
            dataframes[timeline_to_rerank].iloc[idx][["label", "text"]],
        ]
    )
    return user_statuses_df, user_statuses_embeddings


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
def _(layout, mo, rerank_form):
    mo.stop(rerank_form.value is None)

    tag_form = layout.create_tag_form("Local file", "retrogaming")

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

        **NOTE: a couple of changes have been applied for the sake of having a functional demo:**

        1. Posts are not actually your own (see above).

        2. The word(s) that you enter below will be used to filter the existing posts in the
        (synthetic) public timeline, rather than running a new tag search on the mastodon server.
        This allows you to still get meaningful posts back without having to connect to an instance.

        Some example search terms you could use: `#AI`, `bouldering`, `books`, `scifi`, `retrogaming`, `movies`.
        If a search term is not found, you will simply see no results.
        """),
            tag_form,
        ]
    )
    return (tag_form,)


@app.cell
def _(
    dataframes,
    embedding_service,
    mo,
    np,
    tag_form,
    user_statuses_df,
    user_statuses_embeddings,
):
    tag_name = tag_form.value

    tag_posts_df = dataframes["public"][
        dataframes["public"]["text"].str.contains(tag_name)
    ]
    tag_posts_embeddings = embedding_service.calculate_embeddings(tag_posts_df["text"])

    # calculate the re-ranking index
    my_idx = np.flip(
        np.argsort(
            np.sum(np.dot(tag_posts_embeddings, user_statuses_embeddings.T), axis=0)
        )
    )
    # let us also show the similarity scores used to calculate the index
    user_statuses_df["scores"] = np.sum(
        np.dot(tag_posts_embeddings, user_statuses_embeddings.T), axis=0
    )

    mo.vstack(
        [
            mo.md(
                f"### Your own posts, re-ranked according to their similarity to posts in {tag_name}"
            ),
            user_statuses_df.iloc[my_idx][["text", "scores"]],
        ]
    )
    # my_posts_df[['text', 'scores']]
    return


@app.cell
def _(EmbeddingService, mo, pickle, time):
    def load_dataframes(data_file):
        dataframes = None
        print(f"Loading cached dataframes from {data_file}")
        try:
            with open(data_file, "rb") as f:
                dataframes = pickle.load(f)
        except FileNotFoundError:
            print(f"File {data_file} not found.")

        return dataframes

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

    return build_cache_embeddings, load_dataframes


if __name__ == "__main__":
    app.run()
