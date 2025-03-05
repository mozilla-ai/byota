import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    # Uncomment this code if you want to run the notebook on marimo cloud
    # import micropip
    # await micropip.install("Mastodon.py")
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

    from byota.embeddings import (
        EmbeddingService,
        LLamafileEmbeddingService,
        OllamaEmbeddingService
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
        mo,
        pd,
        pickle,
        requests,
        time,
    )


@app.cell
def _(Path):
    # internal variables

    # client and user credentials filenames
    clientcred_filename = "secret_clientcred.txt"
    usercred_filename = "secret_usercred.txt"

    # dump files for offline mode
    paginated_data_file = "dump_paginated_data_.pkl"
    dataframes_data_file = "dump_dataframes_.pkl"
    embeddings_data_file = "dump_embeddings_.pkl"

    app_registered = True if Path(clientcred_filename).is_file() else False
    return (
        app_registered,
        clientcred_filename,
        dataframes_data_file,
        embeddings_data_file,
        paginated_data_file,
        usercred_filename,
    )


@app.cell
def _(app_registered, mo, reg_form, show_if):
    show_if(not app_registered, reg_form, mo.md("**Your application is registered**"))
    return


@app.cell
def _(
    app_registered,
    byota_mastodon,
    clientcred_filename,
    invalid_form,
    mo,
    reg_form,
):
    if not app_registered:
        mo.stop(invalid_form(reg_form), mo.md("**Invalid values provided in the registration form**"))

        byota_mastodon.register_app(
            reg_form.value['application_name'],
            reg_form.value['api_base_url'],
            clientcred_filename
        )
    return


@app.cell
def _(auth_form):
    auth_form
    return


@app.cell
def _(
    LLamafileEmbeddingService,
    OllamaEmbeddingService,
    auth_form,
    byota_mastodon,
    clientcred_filename,
    invalid_form,
    mo,
    timelines_dict,
    usercred_filename,
):
    # check for anything invalid in the form
    mo.stop(invalid_form(auth_form),
            mo.md("**Submit the form to continue.**"))

    # login (and break if that does not work)
    mastodon_client = byota_mastodon.login(clientcred_filename,
                                           usercred_filename,
                                           auth_form.value.get("login"),
                                           auth_form.value.get("pw")
                                          )
    mo.stop(mastodon_client is None,
            mo.md("**Authentication error.**"))

    # instatiate an embedding service (and break if it does not work)
    if auth_form.value["emb_server"]=="llamafile":
        embedding_service = LLamafileEmbeddingService(
            auth_form.value["emb_server_url"]
        )
    else:
        embedding_service = OllamaEmbeddingService(
            auth_form.value["emb_server_url"],
            auth_form.value["emb_server_model"]
        )

    mo.stop(
        not embedding_service.is_working(),
        mo.md(f"**Cannot access {auth_form.value['emb_server']} embedding server.**"),
    )

    # collect the names of the timelines we want to download from
    timelines = []
    for k in timelines_dict.keys():
        if auth_form.value[k]:
            tl_string = timelines_dict[k]
            if tl_string in ["tag", "list"]:
                tl_string += f'/{auth_form.value[f"{k}_txt"]}'
            timelines.append(tl_string)

    # set offline mode
    offline_mode = auth_form.value["offline_mode"]

    # choose what to read from cache
    cached_timelines =  offline_mode
    cached_dataframes = offline_mode
    cached_embeddings = offline_mode
    return (
        cached_dataframes,
        cached_embeddings,
        cached_timelines,
        embedding_service,
        k,
        mastodon_client,
        offline_mode,
        timelines,
        tl_string,
    )


@app.cell
def _(mo):
    mo.md(r"""# Getting data from my Mastodon account...""")
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
    paginated_data = build_cache_paginated_data(mastodon_client,
                                                timelines,
                                                cached_timelines,
                                                paginated_data_file)
    mo.stop(paginated_data is None, mo.md(f"**Issues connecting to Mastodon**"))


    dataframes = build_cache_dataframes(paginated_data,
                                         cached_dataframes,
                                         dataframes_data_file)

    mo.stop(paginated_data is None, mo.md(f"**Issues connecting to Mastodon**"))
    return dataframes, paginated_data


@app.cell
def _(mo):
    mo.md("""# My timeline(s)""")
    return


@app.cell
def _(
    build_cache_embeddings,
    cached_embeddings,
    dataframes,
    embedding_service,
    embeddings_data_file,
):
    # calculate embeddings
    embeddings = build_cache_embeddings(embedding_service,
                                        dataframes,
                                        cached_embeddings,
                                        embeddings_data_file)
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
            end_idx+=len(embeddings[kk])
            df = dataframes[kk]
            df["x"] = all_projections[start_idx:end_idx, 0]
            df["y"] = all_projections[start_idx:end_idx, 1]
            df["label"] = kk
            dfs.append(df)
            start_idx=end_idx

        return pd.concat(dfs, ignore_index=True), all_embeddings


    df_, all_embeddings = tsne(dataframes, embeddings, perplexity=16)

    chart = mo.ui.altair_chart(
        alt.Chart(df_, title="Timeline Visualization", height=500)
        .mark_point()
        .encode(
            x="x",
            y="y",
            color="label"
        )
    )
    return all_embeddings, chart, df_, np, tsne


@app.cell
def _(chart, mo):
    mo.vstack(
        [
            chart,
            chart.value[["id", "label", "text"]] if len(chart.value) > 0 else chart.value,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""# Timeline search""")
    return


@app.cell
def _(query_form):
    query_form
    return


@app.cell
def _(SearchService, all_embeddings, df_, embedding_service, query_form):
    search_service = SearchService(all_embeddings, embedding_service)
    indices = search_service.most_similar_indices(query_form.value)
    df_.iloc[indices][['label','text']]
    return indices, search_service


@app.cell
def _(all_embeddings, np, query_form, search_service):
    import matplotlib.pyplot as plt

    mse = search_service.most_similar_embeddings(query_form.value)
    diff_small = mse[0]-mse[1]
    diff_mid = mse[0]-mse[4]
    diff_large = mse[0]-all_embeddings[42]

    plt.rcParams["figure.figsize"] = (20,3)
    plt.plot(diff_large)
    plt.plot(diff_small)
    plt.legend([f"Diff with a random embedding (norm={np.linalg.norm(diff_large):.2f})", 
                f"Diff with a similar embedding (norm={np.linalg.norm(diff_small):.2f})"])

    plt.show()
    return diff_large, diff_mid, diff_small, mse, plt


@app.cell
def _(rerank_form):
    rerank_form
    return


@app.cell
def _(byota_mastodon, mastodon_client, mo, rerank_form):
    # check for anything invalid in the form
    mo.stop(rerank_form.value is None,
            mo.md("**Submit the form to continue.**"))

    user_statuses = byota_mastodon.get_paginated_statuses(mastodon_client,
                        max_pages=rerank_form.value["num_user_status_pages"],
                        exclude_reblogs=rerank_form.value["exclude_reblogs"])
    return (user_statuses,)


@app.cell
def _(mo):
    mo.md("""## Your statuses:""")
    return


@app.cell
def _(get_compact_data, pd, user_statuses):
    user_statuses_df = pd.DataFrame(get_compact_data(user_statuses),
                                    columns=["id", "text"])
    user_statuses_df
    return (user_statuses_df,)


@app.cell
def _(mo):
    mo.md("""## Your re-ranked timeline:""")
    return


@app.cell
def _(
    dataframes,
    embedding_service,
    embeddings,
    np,
    rerank_form,
    time,
    user_statuses_df,
):
    # calculate embeddings for user statuses
    user_statuses_embeddings = embedding_service.calculate_embeddings(user_statuses_df["text"])

    timeline_to_rerank = rerank_form.value["timeline_to_rerank"]

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
                    np.dot(
                        user_statuses_embeddings,
                        embeddings[timeline_to_rerank].T), axis=0)))

    print(time.time()-rerank_start_time)

    # return the statuses sorted by that index
    dataframes[timeline_to_rerank].iloc[idx][['label','text']]
    return (
        idx,
        rerank_start_time,
        timeline_to_rerank,
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
def _(mo):
    mo.md("""### My own posts, re-ranked according to their similarity to posts in tag/gopher""")
    return


@app.cell
def _(
    byota_mastodon,
    embedding_service,
    get_compact_data,
    mastodon_client,
    np,
    pd,
):
    my_posts = byota_mastodon.get_paginated_statuses(mastodon_client,
                        max_pages=10,
                        exclude_reblogs=True, exclude_replies=True)
    my_posts_df = pd.DataFrame(get_compact_data(my_posts),
                                    columns=["id", "text"])
    my_posts_embeddings = embedding_service.calculate_embeddings(my_posts_df["text"])

    ds_posts = byota_mastodon.get_paginated_data(mastodon_client, "tag/gopher", max_pages=1)
    ds_posts_df = pd.DataFrame(get_compact_data(ds_posts),
                                    columns=["id", "text"])
    ds_posts_embeddings = embedding_service.calculate_embeddings(ds_posts_df["text"])


    my_idx = np.flip(np.argsort(np.sum(np.dot(
                        ds_posts_embeddings,
                        my_posts_embeddings.T), axis=0)))
    my_posts_df["scores"]= np.sum(np.dot(ds_posts_embeddings,my_posts_embeddings.T), axis=0)
    my_posts_df.iloc[my_idx][['text', 'scores']]
    # my_posts_df[['text', 'scores']]
    return (
        ds_posts,
        ds_posts_df,
        ds_posts_embeddings,
        my_idx,
        my_posts,
        my_posts_df,
        my_posts_embeddings,
    )


@app.cell
def _(mo):
    # Create the Configuration form

    auth_form = (
        mo.md(
            """
        # Configuration
        **Mastodon Credentials**

        {login}         {pw}

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
            login=mo.ui.text(label="Login:"),
            pw=mo.ui.text(label="Password:", kind="password"),
            tl_home=mo.ui.checkbox(label="Home", value=True),
            tl_local=mo.ui.checkbox(label="Local"),
            tl_public=mo.ui.checkbox(label="Public"),
            tl_hashtag=mo.ui.checkbox(label="Hashtag"),
            tl_list=mo.ui.checkbox(label="List"),
            tl_hashtag_txt=mo.ui.text(),
            tl_list_txt=mo.ui.text(),
            emb_server=mo.ui.radio(label="Server type:",
                                   options=["llamafile", "ollama"],
                                   value="llamafile",
                                   inline=True
                                  ),
            emb_server_url=mo.ui.text(
                label="Embedding server URL:",
                value="http://localhost:8080/embedding",
                full_width=True
            ),
            emb_server_model=mo.ui.text(
                label="Embedding server model:",
                value="all-minilm"
            ),
            offline_mode=mo.ui.checkbox(label="Run in offline mode"),
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
    return auth_form, invalid_form, timelines_dict


@app.cell
def _(mo):
    # Create a form for timeline re-ranking
    rerank_form = (
        mo.md(
            """
        # Timeline re-ranking
        **User statuses**

        {num_user_status_pages}    {exclude_reblogs}

        **Timelines**

        {timeline_to_rerank}
    """
        )
        .batch(
            num_user_status_pages=mo.ui.slider(start=1, stop=20,
                                               label="Number of pages to load",
                                               value=1),
            timeline_to_rerank=mo.ui.radio(options=["home", "local", "public"], value="home"),
            exclude_reblogs=mo.ui.checkbox(label="Exclude reblogs")
        )
        .form(show_clear_button=True, bordered=True)
    )
    return (rerank_form,)


@app.cell
def _(mo):
    # Create a registration form

    default_api_base_url = "https://your.instance.url"

    reg_form = (
        mo.md(
            """
        # App Registration
        **Register your application**

        {application_name}
        {api_base_url}

    """
        )
        .batch(
            application_name=mo.ui.text(
                label="Application name:",
                value="my_timeline_algorithm",
                full_width=True
            ),
            api_base_url=mo.ui.text(
                label="Mastodon instance API base URL:",
                value=default_api_base_url,
                full_width=True
            ),
        )
        .form(show_clear_button=True, bordered=True)
    )

    def invalid_reg_form(reg_form):
        """A reg form is invalid if the URL is the default one"""
        if reg_form.value is None:
            return True

        for k in reg_form.value.keys():
            if reg_form.value[k] is None or reg_form.value[k]=="":
                return True

        if reg_form.value['api_base_url']==default_api_base_url:
            return True

        return False


    def show_if(condition: bool, if_true, if_false):
        if condition:
            return if_true
        else:
            return if_false
    return default_api_base_url, invalid_reg_form, reg_form, show_if


@app.cell
def _(mo):
    query_form = mo.ui.text(
        value="42",
        label="Enter a status id or some free-form text to find the most similar statuses:\n",
        full_width=True,
    )
    return (query_form,)


@app.cell
def _(BeautifulSoup, EmbeddingService, byota_mastodon, pd, pickle, time):
    def build_cache_paginated_data(mastodon_client,
                                   timelines: list,
                                   cached: bool,
                                   paginated_data_file: str) -> dict[str,any]:
        """Given a list of timeline names and a mastodon client,
        use the mastodon client to get paginated data from each
        and return a dictionary that contains, for each key, all
        the retrieved data.
        If cached==True, the `paginated_data_file` file will be loaded.
        """
        if not cached:
            paginated_data = {}
            for tl in timelines:
                paginated_data[tl] = byota_mastodon.get_paginated_data(mastodon_client, tl)
            with open(paginated_data_file, "wb") as f:
                pickle.dump(paginated_data, f)

        else:
            print(f"Loading cached paginated data from {paginated_data_file}")
            with open(paginated_data_file, "rb") as f:
                paginated_data = pickle.load(f)

        return paginated_data


    def build_cache_dataframes(paginated_data: dict[str, any],
                               cached: bool,
                               dataframes_data_file: str) -> dict[str, any]:
        """Given a dictionary with paginated data from different timelines,
        return another dictionary that contains, for each timeline, a compact
        pandas DataFrame of (id, text) pairs.
        If cached==True, the `dataframes_data_file` file will be loaded.
        """
        if not cached:
            dataframes = {}
            for k in paginated_data:
                dataframes[k] = pd.DataFrame(
                    get_compact_data(paginated_data[k]), 
                    columns=["id", "text"]
                )
            with open(dataframes_data_file, "wb") as f:
                pickle.dump(dataframes, f)
        else:
            print(f"Loading cached dataframes from {dataframes_data_file}")
            with open(dataframes_data_file, "rb") as f:
                dataframes = pickle.load(f)

        return dataframes


    def build_cache_embeddings(embedding_service: EmbeddingService,
                               dataframes: dict[str, any],
                               cached: bool,
                               embeddings_data_file: str) -> dict[str, any]:
        """Given a dictionary with dataframes from different timelines,
        return another dictionary that contains, for each timeline, the
        respective embeddings calculated with the provided embedding service.
        If cached==True, the `embeddings_data_file` file will be loaded.
        """
        if not cached:
            embeddings = {}
            for k in dataframes:
                print(f"Embedding statuses from timeline: {k}")
                tt_ = time.time()
                embeddings[k] = embedding_service.calculate_embeddings(dataframes[k]["text"])
                print(time.time() - tt_)
            with open(embeddings_data_file, "wb") as f:
                pickle.dump(embeddings, f)
        else:
            print(f"Loading cached embeddings from {embeddings_data_file}")
            with open(embeddings_data_file, "rb") as f:
                embeddings = pickle.load(f)

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
