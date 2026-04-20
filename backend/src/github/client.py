import logging
import re
import time

from github import Github, GithubException
from github.Repository import Repository

import config

logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s]+", "-", text.strip())
    return text[:50].rstrip("-")


class GitHubClient:
    def __init__(self):
        self.g = Github(config.GITHUB_TOKEN)
        self.username = config.GITHUB_USERNAME
        self.user = self.g.get_user(self.username)

    def get_issue_details(self, owner: str, repo_name: str, issue_number: int) -> dict:
        repo = self.g.get_repo(f"{owner}/{repo_name}")
        issue = repo.get_issue(issue_number)
        return {
            "title": issue.title,
            "body": issue.body or "",
            "labels": [lbl.name for lbl in issue.labels],
            "url": issue.html_url,
        }

    def fork_repo(self, owner: str, repo_name: str) -> Repository:
        source = self.g.get_repo(f"{owner}/{repo_name}")

        # Check if already forked
        try:
            forked = self.g.get_repo(f"{self.username}/{repo_name}")
            logger.info(f"Fork already exists: {forked.full_name}")
            return forked
        except GithubException:
            pass

        logger.info(f"Forking {owner}/{repo_name}...")
        forked = self.user.create_fork(source)

        # Wait for fork with exponential backoff (2s, 4s, 8s … up to 30s, max ~3 min)
        for attempt in range(12):
            wait = min(2 ** (attempt + 1), 30)
            time.sleep(wait)
            try:
                forked = self.g.get_repo(f"{self.username}/{repo_name}")
                if forked.fork:
                    logger.info(f"Fork ready: {forked.clone_url}")
                    return forked
            except GithubException:
                pass

        raise RuntimeError(f"Fork of {owner}/{repo_name} did not become ready in time")

    def make_branch_name(self, issue_number: int, issue_title: str) -> str:
        slug = _slugify(issue_title)
        return f"fix/issue-{issue_number}-{slug}"

    def create_pull_request(
        self,
        source_repo: Repository,
        fork_repo: Repository,
        branch: str,
        issue_number: int,
        issue_title: str,
        fix_summary: str,
    ) -> str:
        pr_title = f"fix: {issue_title}"

        pr_body = (
            f"## Summary\n\n"
            f"{fix_summary}\n\n"
            f"---\n\n"
            f"closes #{issue_number}"
        )

        head = f"{self.username}:{branch}"

        try:
            pr = source_repo.create_pull(
                title=pr_title,
                body=pr_body,
                head=head,
                base=source_repo.default_branch,
            )
            logger.info(f"PR created: {pr.html_url}")
            return pr.html_url
        except GithubException as e:
            # A PR already exists for this branch — find and return it instead of failing.
            if e.status == 422 and "already exists" in str(e):
                open_prs = source_repo.get_pulls(state="open", head=head)
                for pr in open_prs:
                    logger.info(f"PR already exists: {pr.html_url}")
                    return pr.html_url
                # PR not found in open state — may have been closed or merged
                logger.warning(f"PR reported as existing but not found open for head={head}")
            logger.error(f"Failed to create PR: {e}")
            raise

    def find_existing_pr(self, owner: str, repo_name: str, issue_number: int) -> str | None:
        """Return URL of any open/merged PR Workman already opened for this issue, or None."""
        try:
            results = self.g.search_issues(
                f"repo:{owner}/{repo_name} is:pr author:{self.username} #{issue_number}"
            )
            for pr in results:
                return pr.html_url
        except Exception as e:
            logger.warning(f"Could not check existing PRs for {owner}/{repo_name}#{issue_number}: {e}")
        return None

    def get_clone_url(self, repo: Repository) -> str:
        """Returns a clean HTTPS URL with no embedded credentials."""
        return f"https://github.com/{repo.full_name}.git"
