import mastodon.return_types
import mastodon.types_base
from byota.mastodon import get_compact_data
import mastodon


def test_get_compact_data(paginated_data):
    # generate a list of mastodon Status objects from the test json file
    status_list = []
    for masto_status in paginated_data:
        status_list.append(
            mastodon.types_base.try_cast_recurse(
                mastodon.return_types.Status, masto_status
            )
        )

    # get compact data honoring the discoverability tag (we expect 1 status)
    compact_data = get_compact_data([status_list])
    assert len(compact_data) == 1

    # get compact data without honoring the discoverability tag (we expect 2 statuses)
    compact_data = get_compact_data([status_list], honor_discoverable=False)
    assert len(compact_data) == 2
