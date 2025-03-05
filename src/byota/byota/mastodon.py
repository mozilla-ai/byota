import mastodon
from pathlib import Path
import marimo as mo

# -- Mastodon ----------------------------------------------------------------

def register_app(app_name: str,
                 api_base_url: str,
                 clientcred_filename:str,
                 ):
    
    mastodon.Mastodon.create_app(app_name,
                                 api_base_url=api_base_url,
                                 to_file=clientcred_filename
                                 )


def login(clientcred_filename: str,
          usercred_filename: str,
          login: str,
          password: str):
    """Checks if client credentials are available and logs user in."""

    usercred_file = Path(usercred_filename)
    clientcred_file = Path(clientcred_filename)
    mastodon_client = None

    try:
        if not usercred_file.is_file():
            # if client credentials are not available, ask to register app first
            if not clientcred_file.is_file():
                print(f"{clientcred_file} not found: you need to register your application first.")
                return mastodon_client

            # authenticate app
            mastodon_client = mastodon.Mastodon(
                client_id=clientcred_filename
            )

            # log in / save access token
            mastodon_client.log_in(login, password, to_file=usercred_filename)

        # if user access token is available, authenticate with it
        mastodon_client = mastodon.Mastodon(
            access_token=usercred_filename,
        )

    except mastodon.errors.MastodonIllegalArgumentError as e:
        print(f"Mastodon auth error: {e}")
        return None

    return mastodon_client


def get_paginated_data(mastodon_client: mastodon.Mastodon, timeline_type: str, max_pages: int = 40):
    """Gets paginated statuses from one of the following timelines:
       `home`, `local`, `public`, `tag/hashtag` or `list/id`.

    See https://mastodonpy.readthedocs.io/en/stable/07_timelines.html
    and https://docs.joinmastodon.org/methods/timelines/#home
    """

    tl = mastodon_client.timeline(timeline_type)

    paginated_data = []
    max_id = None
    i = 1
    with mo.status.progress_bar(total=max_pages,
                                title=f"Timeline: {timeline_type}") as bar:
        while len(tl) > 0 and i <= max_pages:
            print(
                f"Loading page {i}: max_id = {max_id}"
            )
            tl = mastodon_client.timeline(timeline_type,
                max_id=max_id
            )
            if len(tl)>0:
                paginated_data.append(tl)

            max_id=tl._pagination_next.get("max_id")
            bar.update()
            i+=1
    return paginated_data


def get_paginated_statuses(mastodon_client: mastodon.Mastodon,
                           max_pages: int = 1,
                           exclude_replies=False,
                           exclude_reblogs=False):
    """Gets paginated statuses from one of the following timelines:
       `home`, `local`, `public`, `tag/hashtag` or `list/id`.

    See https://mastodonpy.readthedocs.io/en/stable/07_timelines.html
    and https://docs.joinmastodon.org/methods/timelines/#home
    """

    tl = mastodon_client.account_statuses(mastodon_client.me()["id"],
                                          exclude_replies=exclude_replies,
                                          exclude_reblogs=exclude_reblogs)

    paginated_data = []
    max_id = None
    i = 1
    with mo.status.progress_bar(total=max_pages,
                                title=f"Account Statuses (replies={not exclude_replies}, reblogs={not exclude_reblogs})") as bar:
        while len(tl) > 0 and i <= max_pages:
            print(
                f"Loading page {i}: max_id = {max_id}"
            )
            tl = mastodon_client.account_statuses(mastodon_client.me()["id"],
                                                  exclude_replies=exclude_replies,
                                                  exclude_reblogs=exclude_reblogs,
                                                  max_id=max_id)
            if len(tl)>0:
                paginated_data.append(tl)

            if hasattr(tl, "_pagination_next"):
                max_id=tl._pagination_next.get("max_id")
            else:
                print("No more pages available.")
                break
            bar.update()
            i+=1
    return paginated_data
