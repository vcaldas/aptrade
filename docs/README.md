# aptrade docs

Documentation site for [aptrade](https://github.com/vcaldas/aptrade), built with [Astro](https://astro.build) and [Starlight](https://starlight.astro.build).

## 🚀 Project Structure

```
docs/
├── public/
├── src/
│   ├── assets/
│   ├── content/
│   │   └── docs/
│   └── content.config.ts
├── astro.config.mjs
├── package.json
└── tsconfig.json
```

Documentation pages are `.md` or `.mdx` files under `src/content/docs/`. Each file maps to a URL route based on its filename.

## 🧞 Commands

All commands are run from the `docs/` directory:

| Command           | Action                                     |
| :---------------- | :----------------------------------------- |
| `npm install`     | Install dependencies                       |
| `npm run dev`     | Start local dev server at `localhost:4321` |
| `npm run build`   | Build production site to `./dist/`         |
| `npm run preview` | Preview the production build locally       |

## 🚢 Deployment

The docs are automatically deployed to [GitHub Pages](https://vcaldas.github.io/aptrade) on every push to `main` via the `.github/workflows/docs.yml` workflow.
