VERSION ?=

.PHONY: release

release:
	@if [ -z "$(VERSION)" ]; then \
		VERSION=$$(git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//'); \
		if [ -z "$$VERSION" ]; then \
			echo "VERSION is required. Use: make release VERSION=0.1.2"; \
			exit 1; \
		fi; \
	else \
		VERSION="$(VERSION)"; \
	fi; \
	sed -i "s/^version = .*/version = \"$$VERSION\"/" pyproject.toml; \
	uv build
