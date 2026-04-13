"""Campaign subsystem — declared ablations over the existing OTel trace log.

One git-tracked ``campaigns/<name>.yaml`` per campaign. No separate status
log: OTel tags each ``training.fit`` span with ``campaign.cell_id`` when
``GRAPHIDS_CAMPAIGN_CELL`` is set; :func:`cell_statuses` reads the spans back.
Design: ``plans/graphids-campaign-manifest.md``.
"""

from __future__ import annotations

from graphids.campaigns.manifest import (
    Campaign,
    CampaignDefaults,
    Cell,
    cell_statuses,
    load_campaign,
    merged_pipeline_config,
)

__all__ = [
    "Campaign",
    "CampaignDefaults",
    "Cell",
    "cell_statuses",
    "load_campaign",
    "merged_pipeline_config",
]
