import nextVitals from 'eslint-config-next/core-web-vitals';

const config = [
  ...nextVitals,
  {
    ignores: ['.next/**', 'node_modules/**', '.venv/**'],
  },
  {
    rules: {
      // Existing hooks intentionally initiate async loading from effects.
      'react-hooks/set-state-in-effect': 'off',
      'react-hooks/refs': 'off',
    },
  },
];

export default config;
