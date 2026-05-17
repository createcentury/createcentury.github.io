import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'createcentury',
  tagline: '',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://createcentury.github.io',
  baseUrl: '/',

  organizationName: 'createcentury',
  projectName: 'createcentury.github.io',
  trailingSlash: false,

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV',
      crossorigin: 'anonymous',
    },
  ],

  onBrokenLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          path: 'docs',
          routeBasePath: 'docs',
          sidebarPath: './sidebars.ts',
        },
        blog: {
          routeBasePath: 'blog',
          showReadingTime: true,
          blogTitle: 'createcentury',
          blogDescription: 'Personal blog by createcentury',
          postsPerPage: 10,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          editUrl: 'https://github.com/createcentury/createcentury.github.io/tree/main/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    [
      '@docusaurus/plugin-client-redirects',
      {
        redirects: [
          {from: '/', to: '/blog'},
        ],
      },
    ],
    [
      '@docusaurus/plugin-content-blog',
      {
        id: 'personal',
        routeBasePath: 'personal',
        path: './personal',
        blogTitle: 'Personal',
        blogDescription: 'Personal notes (日本語)',
        showReadingTime: true,
        postsPerPage: 10,
        feedOptions: {
          type: ['rss', 'atom'],
          xslt: true,
        },
        remarkPlugins: [remarkMath],
        rehypePlugins: [rehypeKatex],
        onInlineTags: 'warn',
        onInlineAuthors: 'warn',
        onUntruncatedBlogPosts: 'ignore',
      },
    ],
  ],

  themes: [
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true,
        language: ['en', 'ja'],
        indexBlog: true,
        indexDocs: false,
        indexPages: false,
        blogRouteBasePath: 'blog',
      },
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'createcentury',
      logo: {
        alt: 'createcentury',
        src: 'img/logo.svg',
        href: '/blog',
      },
      items: [
        {to: '/blog', label: 'Blog', position: 'left'},
        {to: '/personal', label: 'Personal', position: 'left'},
        {
          href: 'https://github.com/createcentury',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Links',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/createcentury',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} createcentury.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
