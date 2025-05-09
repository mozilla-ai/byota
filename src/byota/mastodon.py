import mastodon
import marimo as mo
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
        print(f"Mastodon auth error: {e}")
        mastodon_client = None

    return mastodon_client


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
            print(f"Loading page {i}: max_id = {max_id}")
            tl = mastodon_client.timeline(timeline_type, max_id=max_id)
            if len(tl) > 0:
                paginated_data.append(tl)

            bar.update()
            i += 1
            if hasattr(tl, "_pagination_next") and tl._pagination_next is not None:
                max_id = tl._pagination_next.get("max_id")
            else:
                print("No more pages available.")
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
            print(f"Loading page {i}: max_id = {max_id}")
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
                print("No more pages available.")
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
