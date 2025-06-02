import marimo as mo
from byota.config import MastodonAccount


def invalid_form(form):
    """A form (e.g. login) is invalid if it has no value,
    or if any of its keys have no value."""
    if form.value is None:
        return True

    for k in form.value.keys():
        if form.value[k] is None:
            return True

    return False


def create_configuration_form(accounts: list[MastodonAccount]):
    """
    Create a dynamic configuration form based on the loaded accounts.

    Args:
        accounts: list of MastodonAccount objects

    Returns:
        Marimo form object
    """
    # Build markdown template dynamically
    timelines_section = ""
    batch_dict = {}

    for acc in accounts:
        account_name = acc.name

        # Add account section to markdown
        timelines_section += f"\n*{acc.name}*:\n\n"
        timelines_section += f"{{tl_{account_name}_home}} {{tl_{account_name}_local}} {{tl_{account_name}_public}} {{tl_{account_name}_tag}} {{tl_{account_name}_tag_txt}} {{tl_{account_name}_list}} {{tl_{account_name}_list_txt}}\n"

        # Add corresponding UI elements to batch dict
        batch_dict[f"tl_{account_name}_home"] = mo.ui.checkbox(label="Home", value=True)
        batch_dict[f"tl_{account_name}_local"] = mo.ui.checkbox(label="Local")
        batch_dict[f"tl_{account_name}_public"] = mo.ui.checkbox(label="Public")
        batch_dict[f"tl_{account_name}_tag"] = mo.ui.checkbox(label="Hashtag")
        batch_dict[f"tl_{account_name}_list"] = mo.ui.checkbox(label="List")
        batch_dict[f"tl_{account_name}_tag_txt"] = mo.ui.text()
        batch_dict[f"tl_{account_name}_list_txt"] = mo.ui.text()

    # Add the non-timeline fields
    batch_dict.update(
        {
            "emb_server": mo.ui.radio(
                label="Server type:",
                options=["llamafile", "ollama"],
                value="llamafile",
                inline=True,
            ),
            "emb_server_url": mo.ui.text(
                label="Embedding server URL:",
                value="http://localhost:8080/embedding",
                full_width=True,
            ),
            "emb_server_model": mo.ui.text(
                label="Embedding server model:", value="all-minilm"
            ),
            "offline_mode": mo.ui.checkbox(label="Run in offline mode (experimental)"),
        }
    )

    # Create the form
    configuration_form = (
        mo.md(f"""
# Configuration

**Timelines**
{timelines_section}
**Embeddings**

{{emb_server}}

{{emb_server_url}}

{{emb_server_model}}

**Caching**

{{offline_mode}}
""")
        .batch(**batch_dict)
        .form(show_clear_button=True, bordered=True)
    )

    return configuration_form


# Create a form for timeline re-ranking
def create_rerank_form(timeline_names, account_names):
    """
    Create a timeline re-ranking form with dynamic timeline and account options.

    Args:
        timeline_names: list of timeline display names from downloaded data
        account_names: list of account names from configuration

    Returns:
        Marimo form object
    """
    return (
        mo.md(
            """
        # Timeline re-ranking
        **User statuses**

        {user_account}

        {num_user_status_pages}    {exclude_reblogs}

        **Timeline to rerank**

        {timeline_to_rerank}
    """
        )
        .batch(
            user_account=mo.ui.radio(
                label="Download user statuses from:",
                options=account_names,
                value=account_names[0] if account_names else None,
            ),
            num_user_status_pages=mo.ui.slider(
                start=1, stop=20, label="Number of pages to load", value=1
            ),
            timeline_to_rerank=mo.ui.radio(
                label="Timeline to rerank:",
                options=timeline_names,
                value=timeline_names[0] if timeline_names else None,
            ),
            exclude_reblogs=mo.ui.checkbox(label="Exclude reblogs"),
        )
        .form(show_clear_button=True, bordered=True)
    )


def create_query_form():
    query_form = mo.ui.text(
        value="42",
        label="Enter a status id or some free-form text to find the most similar statuses:\n",
        full_width=True,
    )
    return query_form


def create_tag_form(user_account: str, default_value: str = "gopher"):
    tag_form = mo.ui.text(
        value=default_value,
        label=f"Enter a tag name to download from the `{user_account}` account:\n",
    )

    return tag_form
