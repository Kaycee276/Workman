import asyncio
import logging

from github import Github

import config
from .models import DripsIssue

logger = logging.getLogger(__name__)


class DripsWatcher:
    def __init__(self):
        self.g = Github(config.GITHUB_TOKEN)
        self.username = config.GITHUB_USERNAME
        self.watch_orgs = [o.strip().lower() for o in config.WATCH_ORGS.split(",") if o.strip()]
        self.watch_label = (config.WATCH_LABEL or "").strip()

    async def get_assigned_issues(self) -> list[DripsIssue]:
        return await asyncio.to_thread(self._fetch)

    def _build_query(self, org: str | None = None) -> str:
        parts = [f"is:issue is:open assignee:{self.username}"]
        if org:
            parts.append(f"org:{org}")
        if self.watch_label:
            parts.append(f'label:"{self.watch_label}"')
        return " ".join(parts)

    def _fetch(self) -> list[DripsIssue]:
        issues: list[DripsIssue] = []
        seen: set[str] = set()

        queries = [self._build_query(org) for org in self.watch_orgs] if self.watch_orgs else [self._build_query()]

        for query in queries:
            logger.info(f"GitHub search: {query}")
            for gh_issue in self.g.search_issues(query):
                issue = self._convert(gh_issue)
                if issue and issue.id not in seen:
                    seen.add(issue.id)
                    issues.append(issue)

        logger.info(f"Found {len(issues)} assigned issue(s)")
        return issues

    def _convert(self, gh_issue) -> DripsIssue | None:
        try:
            repo = gh_issue.repository
            return DripsIssue(
                id=f"{repo.owner.login}/{repo.name}#{gh_issue.number}",
                title=gh_issue.title,
                description=gh_issue.body or "",
                github_issue_url=gh_issue.html_url,
                github_repo_url=repo.html_url,
                repo_owner=repo.owner.login,
                repo_name=repo.name,
                issue_number=gh_issue.number,
                labels=[lbl.name for lbl in gh_issue.labels],
            )
        except Exception as e:
            logger.warning(f"Could not convert issue: {e}")
            return None
