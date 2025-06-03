#!/usr/bin/env python3
"""
Template Sync Python Library

This module provides functionality for connecting repositories to appropriate template repositories:
1. Identifying appropriate templates based on the majority language of a repository
2. Running template sync GitHub actions (creating PRs)
3. Approving and merging PRs with the template-sync label

Requirements: 
- Python 3.6+
- GitHub CLI (gh)
- PyYAML and ruamel.yaml for YAML handling
- requests for API calls

Usage as a library:
    from template_sync_python import (
        get_repository_majority_language, 
        find_matching_template_repo,
        ensure_template_sync_workflow,
        create_pr_and_merge,
        process_repositories
    )
    
    # Process a specific repository
    language = get_repository_majority_language("org/repo")
    template = find_matching_template_repo("org", language)
    ensure_template_sync_workflow("org/repo", template, "org")
    create_pr_and_merge("org/repo", "user", "approver")
    
    # Or process all repositories in an organization
    process_repositories(org="org", gh_user="user", approver_user="approver")
"""

import os
import subprocess
import json
import time
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Any
import tempfile
import shutil

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add handler if not already added
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
# Enable debug logging if needed
DEBUG = os.environ.get("DEBUG", "0") == "1"
if DEBUG:
    logger.setLevel(logging.DEBUG)


def debug(message: str) -> None:
    """Print debug messages when DEBUG is enabled."""
    if DEBUG:
        logger.debug(message)


def run_command(command: List[str], check: bool = True, capture_output: bool = True) -> Tuple[int, str, str]:
    """
    Run a shell command and return exit code, stdout, and stderr.
    
    Args:
        command: List of command and arguments
        check: Whether to raise an exception on non-zero exit code
        capture_output: Whether to capture and return stdout/stderr
        
    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    try:
        result = subprocess.run(
            command,
            check=check,
            text=True,
            capture_output=capture_output
        )
        return result.returncode, result.stdout.strip() if result.stdout else "", result.stderr.strip() if result.stderr else ""
    except subprocess.CalledProcessError as e:
        if check:
            logger.error(f"Command failed: {' '.join(command)}")
            logger.error(f"Exit code: {e.returncode}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            raise
        return e.returncode, e.stdout.strip() if e.stdout else "", e.stderr.strip() if e.stderr else ""


def get_repository_majority_language(repo: str) -> Optional[str]:
    """
    Get the majority language used in a repository.
    
    Args:
        repo: Repository name in format 'owner/repo'
        
    Returns:
        The name of the majority language or None if not found
    """
    # Normalize common language variations
    language_normalizations = {
        'C#': 'c#',
        'CSharp': 'c#',
        'Csharp': 'c#',
        'C++': 'c++',
        'Cpp': 'c++',
        'JavaScript': 'javascript',
        'TypeScript': 'typescript',
        'Go': 'go',
        'Golang': 'go',
        'Python': 'python',
        'Ruby': 'ruby',
        'Java': 'java',
        'PHP': 'php',
        'Swift': 'swift',
        'Objective-C': 'objective-c',
        'Rust': 'rust',
    }
    
    _, output, _ = run_command(["gh", "api", f"repos/{repo}/languages"])
    if not output:
        logger.warning(f"Could not determine languages for {repo}")
        return None
    
    languages = json.loads(output)
    if not languages:
        return None
    
    # Find language with highest byte count
    majority_language = max(languages.items(), key=lambda x: x[1])[0]
    
    # Normalize the language name if needed
    normalized_language = language_normalizations.get(majority_language, majority_language.lower())
    
    logger.info(f"Majority language for {repo}: {majority_language} (normalized: {normalized_language})")
    return normalized_language


def find_matching_template_repo(org: str, language: str) -> Optional[str]:
    """
    Find a template repository that matches the given language.
    
    Args:
        org: GitHub organization
        language: Programming language to match
        
    Returns:
        Template repository name or None if not found
    """
    # Get all repositories that are templates
    _, output, _ = run_command([
        "gh", "repo", "list", org,
        "--json", "nameWithOwner,isTemplate,name",
        "--limit", "100"
    ])
    
    repos = json.loads(output)
    
    # Known language mappings and variations
    language_mappings = {
        'c#': ['csharp', 'dotnet', 'dotnet-core'],
        'javascript': ['js', 'node', 'nodejs'],
        'typescript': ['ts', 'node', 'nodejs'],
        'python': ['py', 'python3'],
        'java': ['java', 'springboot', 'spring'],
        'go': ['golang'],
        'ruby': ['rb'],
        'php': ['php'],
        'c++': ['cpp', 'cplusplus'],
        'rust': ['rs'],
    }
    
    # Specific template repositories to prioritize
    specific_templates = {
        'c#': 'github-template-csharp-dotnet-core-repo',
        # Add more specific templates as needed
    }
    
    # If we have a specific template for this language, check if it exists
    language_lower = language.lower()
    if language_lower in specific_templates.keys():
        specific_repo = specific_templates[language_lower]
        for repo in repos:
            if repo.get("name") == specific_repo and repo.get("isTemplate"):
                logger.info(f"Found specific template repository for {language}: {specific_repo}")
                return specific_repo
    
    # Get potential search terms for this language
    search_terms = [language_lower]
    if language_lower in language_mappings:
        search_terms.extend(language_mappings[language_lower])
    
    logger.debug(f"Searching for templates with terms: {search_terms}")
    
    # Find repositories that match any of the search terms
    template_repos = []
    for repo in repos:
        if not repo.get("isTemplate", False):
            continue
            
        repo_name = repo.get("name", "").lower()
        
        # Check if any search term is in the repo name
        for term in search_terms:
            if term in repo_name:
                template_repos.append(repo["name"])
                break
    
    if not template_repos:
        logger.warning(f"No template repository found for {language}")
        return None
    
    # Sort by priority - repositories with the exact language name should come first
    template_repos.sort(key=lambda x: 0 if language_lower in x.lower() else 1)
    
    logger.info(f"Found template repositories for {language}: {template_repos}")
    return template_repos[0]


def create_pr_and_merge(repo: str, gh_user: str, approver_user: str) -> None:
    """
    Run template sync on a repository, then approve and merge the resulting PR.
    
    Args:
        repo: Repository name in format 'owner/repo'
        gh_user: GitHub username to create PRs with
        approver_user: GitHub username to approve and merge PRs
    """
    # Run the template sync GitHub action
    logger.info(f"Running template-sync workflow for {repo}")
    run_command(["gh", "workflow", "run", "template-sync.yml", "-R", repo])
    
    # Wait for the PR to be created (polling)
    logger.info("Waiting for template-sync PR to be created...")
    pr_number = None
    for i in range(10):
        _, output, _ = run_command([
            "gh", "pr", "list", "-R", repo,
            "--state", "open",
            "--label", "template-sync",
            "--json", "number",
            "--jq", ".[0].number"
        ], check=False)
        
        if output.strip():
            pr_number = output.strip()
            break
        
        logger.info(f"Waiting for template-sync PR to be created... ({i+1}/10)")
        time.sleep(10)
    
    # Approve and merge the PR if found
    if pr_number:
        logger.info(f"Approving and merging template-sync PR #{pr_number}")
        
        # Switch to approver user
        run_command(["gh", "auth", "switch", "-u", approver_user])
        
        # Approve and merge
        run_command(["gh", "pr", "review", "--approve", pr_number, "-R", repo])
        run_command(["gh", "pr", "merge", "--auto", "--squash", pr_number, "-R", repo])
        
        # Switch back to original user
        run_command(["gh", "auth", "switch", "-u", gh_user])
    else:
        logger.warning("No template-sync PR found after waiting.")


def ensure_template_sync_workflow(repo: str, template_repo: str, org: str) -> bool:
    """
    Ensures that the repository has a template-sync workflow file pointing to the correct template.
    Creates or updates the workflow file as needed.
    
    Args:
        repo: Repository name in format 'owner/repo'
        template_repo: Template repository name
        org: GitHub organization name
        
    Returns:
        True if successful, False if failed
    """
    import yaml
    from ruamel.yaml import YAML
    
    repo_name = repo.split('/')[-1]
    temp_dir = tempfile.mkdtemp(prefix=f"temp-{repo_name}-")
    
    try:
        # Clone the repository
        logger.info(f"Cloning {repo} to update template-sync workflow")
        run_command(["gh", "repo", "clone", repo, temp_dir, "--", "-q"])
        
        # .releaserc.yaml is at the root of the repository
        workflow_file = os.path.join(temp_dir, ".releaserc.yaml")
        
        # We need to set up two files:
        # 1. .github/workflows/template-sync.yml - the workflow file
        # 2. .releaserc.yaml - at the root of the repo
        
        ryaml = YAML()
        ryaml.preserve_quotes = True
        ryaml.indent(mapping=2, sequence=4, offset=2)
        
        # Set up the workflow file
        workflow_dir = os.path.join(temp_dir, ".github", "workflows")
        os.makedirs(workflow_dir, exist_ok=True)
        workflow_file = os.path.join(workflow_dir, "template-sync.yml")
        
        # Configure the workflow
        workflow_data = None
        if os.path.exists(workflow_file):
            with open(workflow_file, 'r') as f:
                workflow_data = ryaml.load(f)
        else:
            # Create a new workflow file
            workflow_data = {
                'name': 'Template Sync',
                'on': {
                    'schedule': [{'cron': '0 0 * * 0'}],  # Weekly on Sunday
                    'workflow_dispatch': {}
                },
                'jobs': {
                    'template-sync': {
                        'runs-on': 'ubuntu-latest',
                        'steps': [
                            {
                                'name': 'Sync template repository',
                                'uses': 'pstleu/github-workflows/.github/actions/template-sync@v2.14.2',
                                'with': {
                                    'source-repo-path': f"{org}/{template_repo}",
                                    'upstream-branch': 'main',
                                    'pr-labels': 'template-sync',
                                    'pr-title': 'Template Sync'
                                }
                            }
                        ]
                    }
                }
            }
        
        # Update the source repo path
        if 'jobs' in workflow_data and 'template-sync' in workflow_data['jobs']:
            if 'with' not in workflow_data['jobs']['template-sync']:
                workflow_data['jobs']['template-sync']['with'] = {}
                
            workflow_data['jobs']['template-sync']['with']['source-repo-path'] = f"{org}/{template_repo}"
            
            # Ensure we're using the right action version
            workflow_data['jobs']['template-sync']['uses'] = 'pstleu/github-workflows/.github/actions/template-sync@v2.14.2'
        
        # Write workflow file
        with open(workflow_file, 'w') as f:
            ryaml.dump(workflow_data, f)
            
        # Now set up the .releaserc.yaml file at the root
        releaserc_file = os.path.join(temp_dir, ".releaserc.yaml")
        releaserc_data = None
        
        if os.path.exists(releaserc_file):
            try:
                with open(releaserc_file, 'r') as f:
                    releaserc_data = ryaml.load(f)
                # Update template info if needed
                if releaserc_data:
                    # Update any template-specific settings here if needed
                    pass
            except Exception as e:
                logger.warning(f"Could not read .releaserc.yaml: {e}")
                # Create a default one
                releaserc_data = {
                    "template": f"{org}/{template_repo}",
                    "branch": "main"
                }
        else:
            # Create a basic .releaserc.yaml file
            releaserc_data = {
                "template": f"{org}/{template_repo}",
                "branch": "main"
            }
            
        # Write .releaserc.yaml file
        with open(releaserc_file, 'w') as f:
            ryaml.dump(releaserc_data, f)
            
        # Commit and push changes
        branch_name = f"template-sync-update-{int(time.time())}"
        
        # Create a new branch
        os.chdir(temp_dir)
        run_command(["git", "checkout", "-b", branch_name])
        run_command(["git", "add", workflow_file, releaserc_file])
        run_command(["git", "commit", "-m", f"Update template sync configuration to use {template_repo}"])
        run_command(["git", "push", "origin", branch_name])
        
        # Create PR
        pr_url = run_command([
            "gh", "pr", "create",
            "--title", "Update template-sync workflow",
            "--body", f"Automated update of template-sync workflow to use {template_repo}.",
            "--base", "main",
            "--head", branch_name,
            "-R", repo
        ])[1]
        
        logger.info(f"Created PR: {pr_url}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to update template-sync workflow: {str(e)}")
        return False
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def process_repositories(org: str, gh_user: str, approver_user: str, limit: int = 100) -> None:
    """
    Process all repositories in an organization to connect them with appropriate templates.
    
    Args:
        org: GitHub organization name
        gh_user: GitHub username to create PRs with
        approver_user: GitHub username to approve and merge PRs
        limit: Maximum number of repositories to process
    """
    # Get all repos in the organization
    logger.info(f"Getting repositories for {org}...")
    _, output, _ = run_command([
        "gh", "repo", "list", org,
        "--json", "nameWithOwner,isTemplate",
        "--limit", str(limit)
    ])
    
    repos = json.loads(output)
    logger.info(f"Found {len(repos)} repositories")
    
    for repo_info in repos:
        repo = repo_info["nameWithOwner"]
        is_template = repo_info.get("isTemplate", False)
        
        if is_template:
            logger.info(f"[SKIP] {repo} is itself a template")
            continue
        
        logger.info(f"Processing {repo}")
        
        # 1. Determine the majority language and find matching template
        majority_language = get_repository_majority_language(repo)
        if not majority_language:
            logger.warning(f"[SKIP] Could not determine majority language for {repo}")
            continue
        
        template_repo = find_matching_template_repo(org, majority_language)
        if not template_repo:
            logger.warning(f"[SKIP] Could not find matching template for {repo} with language {majority_language}")
            continue
        
        logger.info(f"Selected template repo for {repo}: {template_repo}")
        
        # Ensure the template sync workflow exists and points to the right template
        if not ensure_template_sync_workflow(repo, template_repo, org):
            logger.error(f"Failed to update template-sync workflow for {repo}")
            continue
            
        # 2-3. Run template sync and approve/merge the PR
        create_pr_and_merge(repo, gh_user, approver_user)
    
    logger.info("Processing complete!")


def main():
    """Command-line entry point for the script."""
    parser = argparse.ArgumentParser(description="Template Sync Python Script")
    parser.add_argument("--org", default="pstleu", help="GitHub organization")
    parser.add_argument("--gh-user", default="aairey", help="GitHub user for API calls")
    parser.add_argument("--approver-user", default="svcacct-trimbletl-github", help="GitHub user for approving PRs")
    parser.add_argument("--limit", type=int, default=100, help="Maximum repositories to process")
    args = parser.parse_args()
    
    process_repositories(
        org=args.org,
        gh_user=args.gh_user,
        approver_user=args.approver_user,
        limit=args.limit
    )


if __name__ == "__main__":
    # Configure basic logging when run as a script
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    main()
