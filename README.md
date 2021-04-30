

# Dataset for self-supervised learning. 
We will use only valid event markers or tags to extract sensor data for the stressed class.
We have two indicators of stress: 1) Tag event markers and 2) Survey timeframes. There are 
also changes in the sensor output which are not included in either tag event markers and survey 
timeframe. Hence, we have these combinations for valid sensor data for the stressed class. 

1. There is a tag event within a survey timeframe. Extract one hour of window around the tag event timestamp.
2. There is a tag event but not within a survey timeframe. If there are significant changes in sensors output, then 
   extract one hour of window around the tag event timestamp.
3. There is a survey timeframe but no tag event within that timeframe. If there are significant changes in sensor output, then 
   extract one hour of window based on the survey timeframe.
4. There is no tag event and survey timeframe but there are significant changes in sensor output. Ignore this. 

Furthermore, in survey we have 3 different arousal indicators:
1. more stressed
2. more overwhelm 
3. more anxious

If for any survey timeframe one of the arousal indicator is valid, we consider that survey timeframe valid for
stress class. Also, the time value for each survey timeframe falls within an hour, we will extract sensor data 
within that hour. 



[comment]: <> (# Product Name)

[comment]: <> (> Short blurb about what your product does.)

[comment]: <> ([![NPM Version][npm-image]][npm-url])

[comment]: <> ([![Build Status][travis-image]][travis-url])

[comment]: <> ([![Downloads Stats][npm-downloads]][npm-url])

[comment]: <> (One to two paragraph statement about your product and what it does.)

[comment]: <> (![]&#40;header.png&#41;)

[comment]: <> (## Installation)

[comment]: <> (OS X & Linux:)

[comment]: <> (```sh)

[comment]: <> (npm install my-crazy-module --save)

[comment]: <> (```)

[comment]: <> (Windows:)

[comment]: <> (```sh)

[comment]: <> (edit autoexec.bat)

[comment]: <> (```)

[comment]: <> (## Usage example)

[comment]: <> (A few motivating and useful examples of how your product can be used. Spice this up with code blocks and potentially more screenshots.)

[comment]: <> (_For more examples and usage, please refer to the [Wiki][wiki]._)

[comment]: <> (## Development setup)

[comment]: <> (Describe how to install all development dependencies and how to run an automated test-suite of some kind. Potentially do this for multiple platforms.)

[comment]: <> (```sh)

[comment]: <> (make install)

[comment]: <> (npm test)

[comment]: <> (```)

[comment]: <> (## Release History)

[comment]: <> (* 0.2.1)

[comment]: <> (    * CHANGE: Update docs &#40;module code remains unchanged&#41;)

[comment]: <> (* 0.2.0)

[comment]: <> (    * CHANGE: Remove `setDefaultXYZ&#40;&#41;`)

[comment]: <> (    * ADD: Add `init&#40;&#41;`)

[comment]: <> (* 0.1.1)

[comment]: <> (    * FIX: Crash when calling `baz&#40;&#41;` &#40;Thanks @GenerousContributorName!&#41;)

[comment]: <> (* 0.1.0)

[comment]: <> (    * The first proper release)

[comment]: <> (    * CHANGE: Rename `foo&#40;&#41;` to `bar&#40;&#41;`)

[comment]: <> (* 0.0.1)

[comment]: <> (    * Work in progress)

[comment]: <> (## Meta)

[comment]: <> (Your Name – [@YourTwitter]&#40;https://twitter.com/dbader_org&#41; – YourEmail@example.com)

[comment]: <> (Distributed under the XYZ license. See ``LICENSE`` for more information.)

[comment]: <> ([https://github.com/yourname/github-link]&#40;https://github.com/dbader/&#41;)

[comment]: <> (## Contributing)

[comment]: <> (1. Fork it &#40;<https://github.com/yourname/yourproject/fork>&#41;)

[comment]: <> (2. Create your feature branch &#40;`git checkout -b feature/fooBar`&#41;)

[comment]: <> (3. Commit your changes &#40;`git commit -am 'Add some fooBar'`&#41;)

[comment]: <> (4. Push to the branch &#40;`git push origin feature/fooBar`&#41;)

[comment]: <> (5. Create a new Pull Request)

[comment]: <> (<!-- Markdown link & img dfn's -->)

[comment]: <> ([npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square)

[comment]: <> ([npm-url]: https://npmjs.org/package/datadog-metrics)

[comment]: <> ([npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square)

[comment]: <> ([travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square)

[comment]: <> ([travis-url]: https://travis-ci.org/dbader/node-datadog-metrics)

[comment]: <> ([wiki]: https://github.com/yourname/yourproject/wiki)