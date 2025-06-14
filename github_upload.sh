#!/bin/bash

echo "GitHub Repository Upload Script for FLASH"
echo "========================================"
echo ""
echo "This script will create a GitHub repository and push your code."
echo ""
echo "Prerequisites:"
echo "1. You need a GitHub Personal Access Token with 'repo' scope"
echo "2. Get one at: https://github.com/settings/tokens/new"
echo ""
read -p "Enter your GitHub username: " GITHUB_USERNAME
read -s -p "Enter your GitHub Personal Access Token: " GITHUB_TOKEN
echo ""
read -p "Enter repository name (default: FLASH-startup-success-prediction): " REPO_NAME
REPO_NAME=${REPO_NAME:-FLASH-startup-success-prediction}

echo ""
echo "Creating repository on GitHub..."

# Create repository using GitHub API
RESPONSE=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
  -d '{
    "name": "'$REPO_NAME'",
    "description": "AI-powered startup success prediction platform with 81%+ accuracy using CAMP framework",
    "private": false,
    "has_issues": true,
    "has_projects": true,
    "has_wiki": true
  }' \
  https://api.github.com/user/repos)

# Check if repo was created successfully
if echo "$RESPONSE" | grep -q '"full_name"'; then
  echo "✅ Repository created successfully!"
  
  # Add remote origin
  git remote add origin https://$GITHUB_USERNAME:$GITHUB_TOKEN@github.com/$GITHUB_USERNAME/$REPO_NAME.git
  
  echo "Pushing code to GitHub..."
  git push -u origin main
  
  echo ""
  echo "✅ Success! Your repository is now available at:"
  echo "https://github.com/$GITHUB_USERNAME/$REPO_NAME"
  
else
  echo "❌ Error creating repository:"
  echo "$RESPONSE" | grep '"message"' | cut -d'"' -f4
  echo ""
  echo "Common issues:"
  echo "- Repository name already exists"
  echo "- Invalid token or insufficient permissions"
  echo "- Network connectivity issues"
fi