import { defineConfig } from 'vitepress'

// Built straight into the Worker's assets at /public/docs, served at /docs.
export default defineConfig({
  base: '/docs/',
  outDir: '../public/docs',
  cleanUrls: true,
  lang: 'en-US',
  title: 'Heinrich',
  titleTemplate: ':title — Heinrich docs',
  description: 'Documentation for Heinrich — a model-forensics instrument that measures what a language model computes.',
  appearance: 'force-dark',
  ignoreDeadLinks: true,
  themeConfig: {
    siteTitle: 'heinrich · docs',
    nav: [
      { text: '↩ hcirnieh.com', link: 'https://hcirnieh.com/' },
      { text: 'Observatory', link: 'https://hcirnieh.com/observatory' },
      { text: 'The Book', link: 'https://github.com/asuramaya/heinrich/blob/main/paper/TGUTOS.pdf' },
      { text: 'GitHub', link: 'https://github.com/asuramaya/heinrich' },
    ],
    sidebar: [
      {
        text: 'The instrument',
        items: [
          { text: 'What Heinrich is', link: '/' },
          { text: 'Install & quick start', link: '/guide' },
        ],
      },
      {
        text: 'Using it',
        items: [
          { text: 'The pipeline', link: '/guide' },
          { text: 'CLI reference', link: '/cli' },
          { text: 'MCP (agent surface)', link: '/mcp' },
        ],
      },
      {
        text: 'The format & the edge',
        items: [
          { text: 'The .mri artifact', link: '/artifact' },
          { text: 'Architecture & the Observatory', link: '/architecture' },
        ],
      },
      {
        text: 'The research',
        items: [
          { text: 'Findings', link: '/findings' },
        ],
      },
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/asuramaya/heinrich' },
    ],
    outline: { level: [2, 3], label: 'On this page' },
    search: { provider: 'local' },
    editLink: {
      pattern: 'https://github.com/asuramaya/heinrich/edit/main/docs/:path',
      text: 'Edit / contribute on GitHub',
    },
    footer: {
      message: 'The DB is the single source of truth. Interpretation is left to the reader.',
      copyright: 'heinrich · <a href="https://hcirnieh.com/">hcirnieh.com</a> — read inward',
    },
  },
})
