import mastodon
import marimo as mo
import pandas as pd
import pickle
from loguru import logger
from bs4 import BeautifulSoup

# -- Mastodon ----------------------------------------------------------------


def login(access_token: str, api_base_url: str):
    """Checks if client credentials are available and logs user in."""

    try:
        mastodon_client = mastodon.Mastodon(
            access_token=access_token, api_base_url=api_base_url
        )

        logger.debug(mastodon_client.app_verify_credentials())
    except mastodon.errors.MastodonUnauthorizedError as e:
        logger.error(f"Mastodon auth error: {e}")
        mastodon_client = None

    return mastodon_client


def extract_selected_timelines(form_value):
    """
    Extract selected timelines from marimo form and build a structured list.

    Args:
        form_value: dictionary from configuration_form.value

    Returns:
        List of dictionaries with keys: name, account, timeline
    """
    timelines = []

    for key, value in form_value.items():
        # Only process timeline checkboxes that are checked
        if key.startswith("tl_") and not key.endswith("_txt") and value:
            # Parse the key: tl_accountname_timelinetype
            parts = key.split("_", 2)  # Split into max 3 parts
            if len(parts) == 3:
                _, account_name, timeline_name = parts

                # Handle special cases for tag and list
                if timeline_name in ["tag", "list"]:
                    txt_key = f"{key}_txt"
                    if txt_key in form_value and form_value[txt_key]:
                        timeline_name += f"/{form_value[txt_key]}"
                    else:
                        # Skip if no hashtag/list name provided
                        continue

                timelines.append(
                    {
                        "name": f"{account_name}:{timeline_name}",
                        "account": account_name,
                        "timeline": timeline_name,
                    }
                )

    return timelines


def get_paginated_data(
    mastodon_client: mastodon.Mastodon, timeline_type: str, max_pages: int = 40
):
    """Gets paginated statuses from one of the following timelines:
       `home`, `local`, `public`, `tag/hashtag` or `list/id`.

    See https://mastodonpy.readthedocs.io/en/stable/07_timelines.html
    and https://docs.joinmastodon.org/methods/timelines/#home
    """

    tl = mastodon_client.timeline(timeline_type)

    paginated_data = []
    max_id = None
    i = 1
    with mo.status.progress_bar(
        total=max_pages,
        title=f"Downloading {max_pages} pages of posts from: {timeline_type}",
    ) as bar:
        while len(tl) > 0 and i <= max_pages:
            logger.info(f"Loading page {i}: max_id = {max_id}")
            tl = mastodon_client.timeline(timeline_type, max_id=max_id)
            if len(tl) > 0:
                paginated_data.append(tl)

            bar.update()
            i += 1
            if hasattr(tl, "_pagination_next") and tl._pagination_next is not None:
                max_id = tl._pagination_next.get("max_id")
            else:
                logger.info("No more pages available.")
                break

    return paginated_data


def get_paginated_statuses(
    mastodon_client: mastodon.Mastodon,
    max_pages: int = 1,
    exclude_replies=False,
    exclude_reblogs=False,
):
    """Gets paginated statuses from one of the following timelines:
       `home`, `local`, `public`, `tag/hashtag` or `list/id`.

    See https://mastodonpy.readthedocs.io/en/stable/07_timelines.html
    and https://docs.joinmastodon.org/methods/timelines/#home
    """

    tl = mastodon_client.account_statuses(
        mastodon_client.me()["id"],
        exclude_replies=exclude_replies,
        exclude_reblogs=exclude_reblogs,
    )

    paginated_data = []
    max_id = None
    i = 1
    with mo.status.progress_bar(
        total=max_pages,
        title=f"Account Statuses (replies={not exclude_replies}, reblogs={not exclude_reblogs})",
    ) as bar:
        while len(tl) > 0 and i <= max_pages:
            logger.info(f"Loading page {i}: max_id = {max_id}")
            tl = mastodon_client.account_statuses(
                mastodon_client.me()["id"],
                exclude_replies=exclude_replies,
                exclude_reblogs=exclude_reblogs,
                max_id=max_id,
            )
            if len(tl) > 0:
                paginated_data.append(tl)

            bar.update()
            i += 1
            if hasattr(tl, "_pagination_next") and tl._pagination_next is not None:
                max_id = tl._pagination_next.get("max_id")
            else:
                logger.info("No more pages available.")
                break
    return paginated_data


def get_compact_data(
    paginated_data: list, honor_discoverable: bool = True
) -> list[tuple[int, str]]:
    """
    Extract compact (id, text) pairs from a paginated list of statuses.
    Honor the author's `discoverable` tag and add the status only if the
    value is True.
    """

    compact_data = []
    for page in paginated_data:
        for toot in page:
            # skip the post if the account has discoverable == False
            if honor_discoverable and not toot.account.discoverable:
                continue
            id = toot.id
            cont = toot.content
            if toot.reblog:
                id = toot.reblog.id
                cont = toot.reblog.content
            soup = BeautifulSoup(cont, features="html.parser")
            # print(f"{id}: {soup.get_text()}")
            compact_data.append((id, soup.get_text()))
    return compact_data


def download_to_dataframes(
    mastodon_clients: dict, timelines: list, cached: bool, dataframes_data_file: str
) -> dict[str, any]:
    """Given a list of timeline dictionaries and a dictionary of mastodon clients,
    use the appropriate mastodon client to get paginated data from each timeline
    and return a dictionary that contains, for each timeline name, all
    the retrieved data.
    If cached==True, the `dataframes_data_file` file will be directly loaded without
    downloading anything.

    Args:
        mastodon_clients: dict mapping account names to client objects
        timelines: list of dicts with keys: name, account, timeline
        cached: whether to use cached data
        paginated_data_file: path to cache file (saved just for inspection)
        dataframes_data_file: path to cache file
    """

    if not cached:
        dataframes = {}
        for tl_info in timelines:
            account_name = tl_info["account"]
            timeline_name = tl_info["timeline"]
            display_name = tl_info["name"]

            if account_name in mastodon_clients:
                # use the already logged-in client for this account
                client = mastodon_clients[account_name]
                pdata = get_paginated_data(client, timeline_name)
            else:
                logger.warning(f"No client found for account '{account_name}'")
                continue

            # build a simple dataframe with just id and text from the paginated data
            df = pd.DataFrame(
                get_compact_data(pdata),
                columns=["id", "text"],
            )

            if len(df) == 0:
                logger.warning(f"No valid posts found for '{display_name}'")
                continue

            dataframes[display_name] = df

        with open(dataframes_data_file, "wb") as f:
            pickle.dump(dataframes, f)

    else:
        logger.info(f"Loading cached dataframes from {dataframes_data_file}")
        try:
            with open(dataframes_data_file, "rb") as f:
                dataframes = pickle.load(f)
        except FileNotFoundError:
            logger.error(f"File {dataframes_data_file} not found.")
            return None

    return dataframes
