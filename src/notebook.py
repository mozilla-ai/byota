import marimo

__generated_with = "0.11.13"
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
    import requests
    import time
    import functools
    import altair as alt
    from bs4 import BeautifulSoup
    from sklearn.manifold import TSNE
    import pandas as pd
    from pathlib import Path
    import json
    import os

    from byota.embeddings import (
        EmbeddingService,
        LLamafileEmbeddingService,
        OllamaEmbeddingService,
    )
    import byota.mastodon as byota_mastodon
    from byota.search import SearchService

    return (
        BeautifulSoup,
        EmbeddingService,
        LLamafileEmbeddingService,
        OllamaEmbeddingService,
        Path,
        SearchService,
        TSNE,
        alt,
        byota_mastodon,
        functools,
        json,
        mo,
        os,
        pd,
        pickle,
        requests,
        time,
    )


@app.cell
def _(Path, byota_mastodon, json, mo, os):
    # internal variables

    # credentials filename
    cred_filename = "auth.json"

    # dump files for offline mode
    paginated_data_file = "dump_paginated_data.pkl"
    dataframes_data_file = "dump_dataframes.pkl"
    embeddings_data_file = "dump_embeddings.pkl"

    # login (and break if that does not work)
    if Path(cred_filename).is_file():
        with Path("auth.json").open() as f:
            credentials = json.load(f)
        mastodon_client = byota_mastodon.login(
            access_token=credentials.get("MASTODON_ACCESS_TOKEN"),
            api_base_url=credentials.get("MASTODON_API_BASE_URL"),
        )
    else:
        mastodon_client = byota_mastodon.login(
            access_token=os.environ.get("MASTODON_ACCESS_TOKEN"),
            api_base_url=os.environ.get("MASTODON_API_BASE_URL"),
        )

    mo.stop(
        mastodon_client is None,
        mo.md("**Authentication error: please check your credentials.**").center(),
    )
    return (
        cred_filename,
        credentials,
        dataframes_data_file,
        embeddings_data_file,
        f,
        mastodon_client,
        paginated_data_file,
    )


@app.cell
def _(configuration_form):
    configuration_form
    return


@app.cell
def _(
    LLamafileEmbeddingService,
    OllamaEmbeddingService,
    configuration_form,
    invalid_form,
    mo,
    timelines_dict,
):
    # check for anything invalid in the form
    mo.stop(
        invalid_form(configuration_form),
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
    timelines = []
    for k in timelines_dict.keys():
        if configuration_form.value[k]:
            tl_string = timelines_dict[k]
            if tl_string in ["tag", "list"]:
                tl_string += f'/{configuration_form.value[f"{k}_txt"]}'
            timelines.append(tl_string)

    # set offline mode
    offline_mode = configuration_form.value["offline_mode"]

    # choose what to read from cache
    cached_timelines = offline_mode
    cached_dataframes = offline_mode
    cached_embeddings = offline_mode
    return (
        cached_dataframes,
        cached_embeddings,
        cached_timelines,
        embedding_service,
        k,
        offline_mode,
        timelines,
        tl_string,
    )


@app.cell
def _(mo, timelines):
    mo.md(f"""
    ###Downloading paginated data from the following timelines: {", ".join(timelines)}
    """).center()
    return


@app.cell
def _(
    build_cache_dataframes,
    build_cache_paginated_data,
    cached_dataframes,
    cached_timelines,
    dataframes_data_file,
    mastodon_client,
    mo,
    paginated_data_file,
    timelines,
):
    paginated_data = build_cache_paginated_data(
        mastodon_client, timelines, cached_timelines, paginated_data_file
    )
    mo.stop(paginated_data is None, mo.md("**Issues getting paginated data**"))

    dataframes = build_cache_dataframes(
        paginated_data, cached_dataframes, dataframes_data_file
    )

    mo.stop(paginated_data is None, mo.md("**Issues building dataframes**"))
    return dataframes, paginated_data


@app.cell
def _(dataframes, mo):
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
def _(TSNE, alt, dataframes, embeddings, mo, pd):
    import numpy as np

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
        alt.Chart(df_, title="Timeline Visualization", height=500)
        .mark_point()
        .encode(x="x", y="y", color="label")
    )
    return all_embeddings, chart, df_, np, tsne


@app.cell
def _(chart, mo):
    mo.vstack(
        [
            chart,
            chart.value[["id", "label", "text"]]
            if len(chart.value) > 0
            else chart.value,
        ]
    )
    return


@app.cell
def _(embeddings, mo, query_form):
    mo.stop(embeddings is None)

    mo.vstack([mo.md("# Timeline search"), query_form])
    return


@app.cell
def _(SearchService, all_embeddings, df_, embedding_service, query_form):
    search_service = SearchService(all_embeddings, embedding_service)
    indices = search_service.most_similar_indices(query_form.value)
    df_.iloc[indices][["label", "text"]]
    return indices, search_service


@app.cell
def _(embeddings, mo, rerank_form):
    mo.stop(embeddings is None)
    rerank_form
    return


@app.cell
def _(
    byota_mastodon,
    dataframes,
    embedding_service,
    embeddings,
    get_compact_data,
    mastodon_client,
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

    # download and calculate embeddings of user statuses
    user_statuses = byota_mastodon.get_paginated_statuses(
        mastodon_client,
        max_pages=rerank_form.value["num_user_status_pages"],
        exclude_reblogs=rerank_form.value["exclude_reblogs"],
    )
    user_statuses_df = pd.DataFrame(
        get_compact_data(user_statuses), columns=["id", "text"]
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
            mo.md("## Your statuses:"),
            user_statuses_df,
            mo.md("## Your re-ranked timeline:"),
            # show statuses sorted by idx
            dataframes[timeline_to_rerank].iloc[idx][["label", "text"]],
        ]
    )
    return (
        idx,
        rerank_start_time,
        timeline_to_rerank,
        user_statuses,
        user_statuses_df,
        user_statuses_embeddings,
    )


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
def _(mo, rerank_form, tag_form):
    mo.stop(rerank_form.value is None)

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
    return


@app.cell
def _(
    byota_mastodon,
    embedding_service,
    get_compact_data,
    mastodon_client,
    mo,
    pd,
    rerank_form,
):
    mo.stop(rerank_form.value is None)

    my_posts = byota_mastodon.get_paginated_statuses(
        mastodon_client, max_pages=10, exclude_reblogs=True, exclude_replies=True
    )
    my_posts_df = pd.DataFrame(get_compact_data(my_posts), columns=["id", "text"])
    my_posts_embeddings = embedding_service.calculate_embeddings(my_posts_df["text"])
    return my_posts, my_posts_df, my_posts_embeddings


@app.cell
def _(
    byota_mastodon,
    embedding_service,
    get_compact_data,
    mastodon_client,
    mo,
    my_posts_df,
    my_posts_embeddings,
    np,
    pd,
    tag_form,
):
    tag_name = f"tag/{tag_form.value}"

    tag_posts = byota_mastodon.get_paginated_data(
        mastodon_client, tag_name, max_pages=1
    )
    tag_posts_df = pd.DataFrame(get_compact_data(tag_posts), columns=["id", "text"])
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
                f"### Your own posts, re-ranked according to their similarity to posts in {tag_name}"
            ),
            my_posts_df.iloc[my_idx][["text", "scores"]],
        ]
    )
    # my_posts_df[['text', 'scores']]
    return my_idx, tag_name, tag_posts, tag_posts_df, tag_posts_embeddings


@app.cell
def _(mo):
    # Create the Configuration form

    configuration_form = (
        mo.md(
            """
        # Configuration

        **Timelines**

        {tl_home} {tl_local} {tl_public}

        {tl_hashtag} {tl_hashtag_txt} {tl_list} {tl_list_txt}

        **Embeddings**

        {emb_server}

        {emb_server_url}

        {emb_server_model}

        **Caching**

        {offline_mode}
    """
        )
        .batch(
            tl_home=mo.ui.checkbox(label="Home", value=True),
            tl_local=mo.ui.checkbox(label="Local"),
            tl_public=mo.ui.checkbox(label="Public"),
            tl_hashtag=mo.ui.checkbox(label="Hashtag"),
            tl_list=mo.ui.checkbox(label="List"),
            tl_hashtag_txt=mo.ui.text(),
            tl_list_txt=mo.ui.text(),
            emb_server=mo.ui.radio(
                label="Server type:",
                options=["llamafile", "ollama"],
                value="llamafile",
                inline=True,
            ),
            emb_server_url=mo.ui.text(
                label="Embedding server URL:",
                value="http://localhost:8080/embedding",
                full_width=True,
            ),
            emb_server_model=mo.ui.text(
                label="Embedding server model:", value="all-minilm"
            ),
            offline_mode=mo.ui.checkbox(label="Run in offline mode (experimental)"),
        )
        .form(show_clear_button=True, bordered=True)
    )

    # a dictionary mapping Timeline UI checkboxes with the respective
    # strings that identify them in the Mastodon API
    timelines_dict = {
        "tl_home": "home",
        "tl_local": "local",
        "tl_public": "public",
        "tl_hashtag": "tag",
        "tl_list": "list",
    }

    def invalid_form(form):
        """A form (e.g. login) is invalid if it has no value,
        or if any of its keys have no value."""
        if form.value is None:
            return True

        for k in form.value.keys():
            if form.value[k] is None:
                return True

        return False

    return configuration_form, invalid_form, timelines_dict


@app.cell
def _(mo):
    # Create a form for timeline re-ranking
    rerank_form = (
        mo.md(
            """
        # Timeline re-ranking
        **User statuses**

        {num_user_status_pages}    {exclude_reblogs}

        **Timeline to rerank**

        {timeline_to_rerank}
    """
        )
        .batch(
            num_user_status_pages=mo.ui.slider(
                start=1, stop=20, label="Number of pages to load", value=1
            ),
            timeline_to_rerank=mo.ui.radio(
                options=["home", "local", "public"], value="home"
            ),
            exclude_reblogs=mo.ui.checkbox(label="Exclude reblogs"),
        )
        .form(show_clear_button=True, bordered=True)
    )
    return (rerank_form,)


@app.cell
def _(mo):
    query_form = mo.ui.text(
        value="42",
        label="Enter a status id or some free-form text to find the most similar statuses:\n",
        full_width=True,
    )
    return (query_form,)


@app.cell
def _(mo):
    tag_form = mo.ui.text(
        value="gopher",
        label="Enter a tag name:\n",
    )
    return (tag_form,)


@app.cell
def _(BeautifulSoup, EmbeddingService, byota_mastodon, mo, pd, pickle, time):
    def build_cache_paginated_data(
        mastodon_client, timelines: list, cached: bool, paginated_data_file: str
    ) -> dict[str, any]:
        """Given a list of timeline names and a mastodon client,
        use the mastodon client to get paginated data from each
        and return a dictionary that contains, for each key, all
        the retrieved data.
        If cached==True, the `paginated_data_file` file will be loaded.
        """
        if not cached:
            paginated_data = {}
            for tl in timelines:
                paginated_data[tl] = byota_mastodon.get_paginated_data(
                    mastodon_client, tl
                )
            with open(paginated_data_file, "wb") as f:
                pickle.dump(paginated_data, f)

        else:
            print(f"Loading cached paginated data from {paginated_data_file}")
            try:
                with open(paginated_data_file, "rb") as f:
                    paginated_data = pickle.load(f)
            except FileNotFoundError:
                print(f"File {paginated_data_file} not found.")
                return None
        return paginated_data

    def build_cache_dataframes(
        paginated_data: dict[str, any], cached: bool, dataframes_data_file: str
    ) -> dict[str, any]:
        """Given a dictionary with paginated data from different timelines,
        return another dictionary that contains, for each timeline, a compact
        pandas DataFrame of (id, text) pairs.
        If cached==True, the `dataframes_data_file` file will be loaded.
        """
        if not cached:
            dataframes = {}
            for k in paginated_data:
                dataframes[k] = pd.DataFrame(
                    get_compact_data(paginated_data[k]), columns=["id", "text"]
                )
            with open(dataframes_data_file, "wb") as f:
                pickle.dump(dataframes, f)
        else:
            print(f"Loading cached dataframes from {dataframes_data_file}")
            try:
                with open(dataframes_data_file, "rb") as f:
                    dataframes = pickle.load(f)
            except FileNotFoundError:
                print(f"File {dataframes_data_file} not found.")
                return None

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

    def get_compact_data(paginated_data: list) -> list[tuple[int, str]]:
        """Extract compact (id, text) pairs from a paginated list of statuses."""
        compact_data = []
        for page in paginated_data:
            for toot in page:
                id = toot.id
                cont = toot.content
                if toot.reblog:
                    id = toot.reblog.id
                    cont = toot.reblog.content
                soup = BeautifulSoup(cont, features="html.parser")
                # print(f"{id}: {soup.get_text()}")
                compact_data.append((id, soup.get_text()))
        return compact_data

    return (
        build_cache_dataframes,
        build_cache_embeddings,
        build_cache_paginated_data,
        get_compact_data,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
