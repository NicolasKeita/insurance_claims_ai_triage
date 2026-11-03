import { defineGlobalStyles, defineTokens } from '@pandacss/dev';

/** Review palettes remain explicit so prepared evidence keeps its existing visual semantics. */
export const reviewTokens = defineTokens({
  colors: {
    review: {
      dark: {
        'page-bg': {
          value: '#10181d',
        },
        surface: {
          value: '#182329',
        },
        'surface-nav': {
          value: '#141e24',
        },
        'surface-subtle': {
          value: '#1d2b31',
        },
        'surface-muted': {
          value: '#25343b',
        },
        'surface-hover': {
          value: '#2a3d42',
        },
        'surface-accent': {
          value: '#1c3433',
        },
        'surface-accent-strong': {
          value: '#2b4842',
        },
        'surface-code': {
          value: '#111b21',
        },
        'surface-photo': {
          value: '#111b21',
        },
        text: {
          value: '#dce6e7',
        },
        ink: {
          value: '#e8eff0',
        },
        muted: {
          value: '#9fb2b4',
        },
        'text-secondary': {
          value: '#bed0ca',
        },
        teal: {
          value: '#88d0c4',
        },
        'teal-strong': {
          value: '#b2e5dc',
        },
        line: {
          value: '#35474d',
        },
        'line-subtle': {
          value: '#2b3c42',
        },
        'line-accent': {
          value: '#47665c',
        },
        'control-border': {
          value: '#5b7077',
        },
        focus: {
          value: '#8edbd0',
        },
        'button-bg': {
          value: '#286b60',
        },
        'button-hover': {
          value: '#327c6f',
        },
        'on-accent': {
          value: '#ffffff',
        },
        'brand-bg': {
          value: '#245e58',
        },
        'positive-bg': {
          value: '#173b32',
        },
        'positive-fg': {
          value: '#a1dec5',
        },
        'positive-border': {
          value: '#345f50',
        },
        'warning-bg': {
          value: '#3a3021',
        },
        'warning-fg': {
          value: '#e9cb8b',
        },
        'warning-border': {
          value: '#685435',
        },
        'danger-bg': {
          value: '#3e2928',
        },
        'danger-fg': {
          value: '#f0b6ac',
        },
        'danger-border': {
          value: '#75473f',
        },
        'info-bg': {
          value: '#1f343f',
        },
        'info-fg': {
          value: '#b2d7ed',
        },
        'info-border': {
          value: '#3d5b6c',
        },
        'procedure-bg': {
          value: '#302b3d',
        },
        'procedure-fg': {
          value: '#d1c3e9',
        },
        'procedure-border': {
          value: '#5c4e76',
        },
        'neutral-bg': {
          value: '#26343a',
        },
        'neutral-fg': {
          value: '#b5c4c4',
        },
        'neutral-border': {
          value: '#45585f',
        },
        'human-bg': {
          value: '#342f22',
        },
        'human-fg': {
          value: '#e0d09f',
        },
        'human-border': {
          value: '#61593c',
        },
        'signal-positive': {
          value: '#8bc3a6',
        },
        'signal-warning': {
          value: '#d6b46a',
        },
        'signal-danger': {
          value: '#e09b8f',
        },
        'signal-info': {
          value: '#88b4c5',
        },
        'meter-track': {
          value: '#263c38',
        },
        'meter-fill': {
          value: '#76b89b',
        },
        backdrop: {
          value: '#030a10b8',
        },
        'drawer-shadow': {
          value: '#00000070',
        },
        'selection-bg': {
          value: '#365f5b',
        },
        'selection-text': {
          value: '#eef8f6',
        },
      },
      light: {
        'page-bg': {
          value: '#f5f7f7',
        },
        surface: {
          value: '#ffffff',
        },
        'surface-nav': {
          value: '#fcfdfc',
        },
        'surface-subtle': {
          value: '#f5f8f3',
        },
        'surface-muted': {
          value: '#f0f4f2',
        },
        'surface-hover': {
          value: '#eaf1ee',
        },
        'surface-accent': {
          value: '#edf3f1',
        },
        'surface-accent-strong': {
          value: '#e8efdf',
        },
        'surface-code': {
          value: '#ffffff',
        },
        'surface-photo': {
          value: '#edf1e6',
        },
        text: {
          value: '#263a3c',
        },
        ink: {
          value: '#183436',
        },
        muted: {
          value: '#5b6e66',
        },
        'text-secondary': {
          value: '#415f4b',
        },
        teal: {
          value: '#14665e',
        },
        'teal-strong': {
          value: '#124c43',
        },
        line: {
          value: '#e0e6e5',
        },
        'line-subtle': {
          value: '#edf0e9',
        },
        'line-accent': {
          value: '#ceddc4',
        },
        'control-border': {
          value: '#7a9086',
        },
        focus: {
          value: '#1b8177',
        },
        'button-bg': {
          value: '#195c52',
        },
        'button-hover': {
          value: '#124c43',
        },
        'on-accent': {
          value: '#ffffff',
        },
        'brand-bg': {
          value: '#164f4a',
        },
        'positive-bg': {
          value: '#edf6ef',
        },
        'positive-fg': {
          value: '#26664d',
        },
        'positive-border': {
          value: '#dcebdd',
        },
        'warning-bg': {
          value: '#fcf5e9',
        },
        'warning-fg': {
          value: '#8a6022',
        },
        'warning-border': {
          value: '#eddfc5',
        },
        'danger-bg': {
          value: '#fbefec',
        },
        'danger-fg': {
          value: '#9b4d43',
        },
        'danger-border': {
          value: '#eed5ce',
        },
        'info-bg': {
          value: '#eff5fa',
        },
        'info-fg': {
          value: '#42677b',
        },
        'info-border': {
          value: '#d6e4ed',
        },
        'procedure-bg': {
          value: '#f5f1f8',
        },
        'procedure-fg': {
          value: '#6c5b81',
        },
        'procedure-border': {
          value: '#e7dff0',
        },
        'neutral-bg': {
          value: '#f5f6f4',
        },
        'neutral-fg': {
          value: '#64736a',
        },
        'neutral-border': {
          value: '#e2e6df',
        },
        'human-bg': {
          value: '#f4f1e5',
        },
        'human-fg': {
          value: '#6b5d2a',
        },
        'human-border': {
          value: '#e8e3cd',
        },
        'signal-positive': {
          value: '#2e7650',
        },
        'signal-warning': {
          value: '#b08532',
        },
        'signal-danger': {
          value: '#ac5748',
        },
        'signal-info': {
          value: '#88a5ad',
        },
        'meter-track': {
          value: '#f1f5eb',
        },
        'meter-fill': {
          value: '#8fa87b',
        },
        backdrop: {
          value: '#18342950',
        },
        'drawer-shadow': {
          value: '#10291825',
        },
        'selection-bg': {
          value: '#cde6df',
        },
        'selection-text': {
          value: '#183436',
        },
      },
    },
  },
  fonts: {
    reviewBody: {
      value: "'Segoe UI',\n    Inter,\n    -apple-system,\n    BlinkMacSystemFont,\n    sans-serif",
    },
  },
});

/** Keep the public custom-property contract for charts and components. */
export const reviewTheme = defineGlobalStyles({
  ":root,\nhtml[data-theme='dark']": {
    colorScheme: 'dark',
    '--page-bg': '{colors.review.dark.page-bg}',
    '--surface': '{colors.review.dark.surface}',
    '--surface-nav': '{colors.review.dark.surface-nav}',
    '--surface-subtle': '{colors.review.dark.surface-subtle}',
    '--surface-muted': '{colors.review.dark.surface-muted}',
    '--surface-hover': '{colors.review.dark.surface-hover}',
    '--surface-accent': '{colors.review.dark.surface-accent}',
    '--surface-accent-strong': '{colors.review.dark.surface-accent-strong}',
    '--surface-code': '{colors.review.dark.surface-code}',
    '--surface-photo': '{colors.review.dark.surface-photo}',
    '--text': '{colors.review.dark.text}',
    '--ink': '{colors.review.dark.ink}',
    '--muted': '{colors.review.dark.muted}',
    '--text-secondary': '{colors.review.dark.text-secondary}',
    '--teal': '{colors.review.dark.teal}',
    '--teal-strong': '{colors.review.dark.teal-strong}',
    '--line': '{colors.review.dark.line}',
    '--line-subtle': '{colors.review.dark.line-subtle}',
    '--line-accent': '{colors.review.dark.line-accent}',
    '--control-border': '{colors.review.dark.control-border}',
    '--focus': '{colors.review.dark.focus}',
    '--button-bg': '{colors.review.dark.button-bg}',
    '--button-hover': '{colors.review.dark.button-hover}',
    '--on-accent': '{colors.review.dark.on-accent}',
    '--brand-bg': '{colors.review.dark.brand-bg}',
    '--positive-bg': '{colors.review.dark.positive-bg}',
    '--positive-fg': '{colors.review.dark.positive-fg}',
    '--positive-border': '{colors.review.dark.positive-border}',
    '--warning-bg': '{colors.review.dark.warning-bg}',
    '--warning-fg': '{colors.review.dark.warning-fg}',
    '--warning-border': '{colors.review.dark.warning-border}',
    '--danger-bg': '{colors.review.dark.danger-bg}',
    '--danger-fg': '{colors.review.dark.danger-fg}',
    '--danger-border': '{colors.review.dark.danger-border}',
    '--info-bg': '{colors.review.dark.info-bg}',
    '--info-fg': '{colors.review.dark.info-fg}',
    '--info-border': '{colors.review.dark.info-border}',
    '--procedure-bg': '{colors.review.dark.procedure-bg}',
    '--procedure-fg': '{colors.review.dark.procedure-fg}',
    '--procedure-border': '{colors.review.dark.procedure-border}',
    '--neutral-bg': '{colors.review.dark.neutral-bg}',
    '--neutral-fg': '{colors.review.dark.neutral-fg}',
    '--neutral-border': '{colors.review.dark.neutral-border}',
    '--human-bg': '{colors.review.dark.human-bg}',
    '--human-fg': '{colors.review.dark.human-fg}',
    '--human-border': '{colors.review.dark.human-border}',
    '--signal-positive': '{colors.review.dark.signal-positive}',
    '--signal-warning': '{colors.review.dark.signal-warning}',
    '--signal-danger': '{colors.review.dark.signal-danger}',
    '--signal-info': '{colors.review.dark.signal-info}',
    '--meter-track': '{colors.review.dark.meter-track}',
    '--meter-fill': '{colors.review.dark.meter-fill}',
    '--backdrop': '{colors.review.dark.backdrop}',
    '--drawer-shadow': '{colors.review.dark.drawer-shadow}',
    '--selection-bg': '{colors.review.dark.selection-bg}',
    '--selection-text': '{colors.review.dark.selection-text}',
  },
  "html[data-theme='light']": {
    colorScheme: 'light',
    '--page-bg': '{colors.review.light.page-bg}',
    '--surface': '{colors.review.light.surface}',
    '--surface-nav': '{colors.review.light.surface-nav}',
    '--surface-subtle': '{colors.review.light.surface-subtle}',
    '--surface-muted': '{colors.review.light.surface-muted}',
    '--surface-hover': '{colors.review.light.surface-hover}',
    '--surface-accent': '{colors.review.light.surface-accent}',
    '--surface-accent-strong': '{colors.review.light.surface-accent-strong}',
    '--surface-code': '{colors.review.light.surface-code}',
    '--surface-photo': '{colors.review.light.surface-photo}',
    '--text': '{colors.review.light.text}',
    '--ink': '{colors.review.light.ink}',
    '--muted': '{colors.review.light.muted}',
    '--text-secondary': '{colors.review.light.text-secondary}',
    '--teal': '{colors.review.light.teal}',
    '--teal-strong': '{colors.review.light.teal-strong}',
    '--line': '{colors.review.light.line}',
    '--line-subtle': '{colors.review.light.line-subtle}',
    '--line-accent': '{colors.review.light.line-accent}',
    '--control-border': '{colors.review.light.control-border}',
    '--focus': '{colors.review.light.focus}',
    '--button-bg': '{colors.review.light.button-bg}',
    '--button-hover': '{colors.review.light.button-hover}',
    '--on-accent': '{colors.review.light.on-accent}',
    '--brand-bg': '{colors.review.light.brand-bg}',
    '--positive-bg': '{colors.review.light.positive-bg}',
    '--positive-fg': '{colors.review.light.positive-fg}',
    '--positive-border': '{colors.review.light.positive-border}',
    '--warning-bg': '{colors.review.light.warning-bg}',
    '--warning-fg': '{colors.review.light.warning-fg}',
    '--warning-border': '{colors.review.light.warning-border}',
    '--danger-bg': '{colors.review.light.danger-bg}',
    '--danger-fg': '{colors.review.light.danger-fg}',
    '--danger-border': '{colors.review.light.danger-border}',
    '--info-bg': '{colors.review.light.info-bg}',
    '--info-fg': '{colors.review.light.info-fg}',
    '--info-border': '{colors.review.light.info-border}',
    '--procedure-bg': '{colors.review.light.procedure-bg}',
    '--procedure-fg': '{colors.review.light.procedure-fg}',
    '--procedure-border': '{colors.review.light.procedure-border}',
    '--neutral-bg': '{colors.review.light.neutral-bg}',
    '--neutral-fg': '{colors.review.light.neutral-fg}',
    '--neutral-border': '{colors.review.light.neutral-border}',
    '--human-bg': '{colors.review.light.human-bg}',
    '--human-fg': '{colors.review.light.human-fg}',
    '--human-border': '{colors.review.light.human-border}',
    '--signal-positive': '{colors.review.light.signal-positive}',
    '--signal-warning': '{colors.review.light.signal-warning}',
    '--signal-danger': '{colors.review.light.signal-danger}',
    '--signal-info': '{colors.review.light.signal-info}',
    '--meter-track': '{colors.review.light.meter-track}',
    '--meter-fill': '{colors.review.light.meter-fill}',
    '--backdrop': '{colors.review.light.backdrop}',
    '--drawer-shadow': '{colors.review.light.drawer-shadow}',
    '--selection-bg': '{colors.review.light.selection-bg}',
    '--selection-text': '{colors.review.light.selection-text}',
  },
});
