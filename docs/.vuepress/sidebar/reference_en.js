module.exports = function () {
  return [{
      title: 'What\'s New',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/en/docs/whatsnew/v0.25.0', 'What’s new in 0.25.0']
      ]
    },
    {
      title: 'Installation',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/en/docs/installation', 'Installation']
      ]
    },
    {
      title: 'Getting started',
      collapsable: true,
      sidebarDepth: 1,
      children: [
        ['/en/docs/getting_started/', 'Index'],
        ['/en/docs/getting_started/overview', 'Package overview'],
        ['/en/docs/getting_started/10min', '10 Minutes to pandas'],
        ['/en/docs/getting_started/basics', 'Essential Basic Functionality'],
        ['/en/docs/getting_started/dsintro', 'Intro to Data Structures'],
        ['/en/docs/getting_started/comparison', 'Comparison with other tools'],
        ['/en/docs/getting_started/tutorials', 'Tutorials']
      ]
    },
    {
      title: 'User Guide',
      collapsable: true,
      sidebarDepth: 1,
      children: [
        ['/en/docs/user_guide/', 'Index'],
        ['/en/docs/user_guide/io', 'IO Tools (Text, CSV, HDF5, …)'],
        ['/en/docs/user_guide/indexing', 'Indexing and Selecting Data'],
        ['/en/docs/user_guide/advanced', 'MultiIndex / advanced indexing'],
        ['/en/docs/user_guide/merging', 'Merge, join, and concatenate'],
        ['/en/docs/user_guide/reshaping', 'Reshaping and pivot tables'],
        ['/en/docs/user_guide/text', 'Working with text data'],
        ['/en/docs/user_guide/missing_data', 'Working with missing data'],
        ['/en/docs/user_guide/categorical', 'Categorical data'],
        ['/en/docs/user_guide/integer_na', 'Nullable integer data type'],
        ['/en/docs/user_guide/visualization', 'Visualization'],
        ['/en/docs/user_guide/computation', 'Computational tools'],
        ['/en/docs/user_guide/groupby', 'Group By: split-apply-combine'],
        ['/en/docs/user_guide/timeseries', 'Time series / date functionality'],
        ['/en/docs/user_guide/timedeltas', 'Time deltas'],
        ['/en/docs/user_guide/style', 'Styling'],
        ['/en/docs/user_guide/options', 'Options and settings'],
        ['/en/docs/user_guide/enhancingperf', 'Enhancing performance'],
        ['/en/docs/user_guide/sparse', 'Sparse data structures'],
        ['/en/docs/user_guide/gotchas', 'Frequently Asked Questions (FAQ)'],
        ['/en/docs/user_guide/cookbook', 'Cookbook']
      ]
    },
    {
      title: 'pandas Ecosystem',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/en/docs/ecosystem', 'pandas Ecosystem']
      ]
    },
    {
      title: 'API Reference',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/en/docs/reference', 'API Reference']
      ]
    },
    {
      title: 'Development',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/en/docs/development/', 'Index']
      ]
    },
    {
      title: 'Release Notes',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/en/docs/whatsnew/', 'Release Notes']
      ]
    }
  ]
}