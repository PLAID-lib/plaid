# Contributing to PLAID Project

Thank you for your interest in contributing to PLAID Project! We welcome contributions from the community to help make this project even better.

## Summary

Everything you need to know if you want to contribute.

- [Summary](#summary)
- [1. Before contributing](#1-before-contributing)
- [2. Reporting Issues](#2-reporting-issues)
- [3. Feature Requests](#3-feature-requests)
- [4. Contributor License Agreement](#4-contributor-license-agreement)

## 1. Before contributing

1). You need to agree with the contributor license agreement detailed below (LICENSE.txt).

2). We follow a set of coding standards to maintain consistency in the codebase. Please adhere to the following guidelines when contributing:

- Use consistent indentation (e.g., 4 spaces) for code.
- Follow the [style guide (PEP 8)](https://www.python.org/dev/peps/pep-0008/): <https://www.python.org/dev/peps/pep-0008/>
- Write clear and concise code with meaningful variable and function names.
- Comment your code to explain complex or non-obvious sections.
- For docstring please follow the [guidelines](https://peps.python.org/pep-0257/): <https://peps.python.org/pep-0257/>
- For tests see : <https://github.com/The-Compiler/pytest-ep2023/>

The only deviations from PEP 8 are the following :

    Function Names and Variable Names should follow the 'snake_case' convention
    Only lowercase letters and underscores between each word ('_')
    Line limit length of 80 characters

**TIP:**
The 'format.sh' script at the root of the project allows you to format all the python files in the project under the PEP8 convention.

3). Please read the contributing rules :

- All changes are integrated by Safran.
- API must be kept simple and flexible.
- Data structures are defined in Containers, we limit the number of advanced
functions in these classes at maximum. Advanced functions are defined
in other files.
- Unitary tests are in Containers classes and functional tests in files
containing more advanced classes. The latter also serve as examples and
client code for the library.
- Each file preferably contain at most one class
- CheckIntegrities must be local (as little imports as possible), aiming to test
only the functions defined in the file, as much as possible. All functions
in the file must be tested, if possible. Please limit the use of functions
from other file to reach that goal, and use small and simple data, otherwise
changes will be painful to propagate if many CheckIntegrities must be updated
as well.
- Favor imports at the beginning of files (to be available for CheckIntegrities
as well without repeat).
- Coverage must be kept higher than 80%.

## 2. Reporting Issues

If you encounter a problem with PLAID Project or have found a bug, please check the [existing issues](https://gitlab.com/drti/plaid/-/issues) to see if it has already been reported. If not, feel free to [open a new issue](https://gitlab.com/drti/plaid/-/issues/new) with the following information:

- A clear and concise description of the issue.
- Steps to reproduce the issue.
- The expected behavior.
- The version of PLAID Project you are using.
- Any relevant screenshots or error messages.

## 3. Feature Requests

If you have an idea for a new feature or enhancement, please [open a new issue](https://gitlab.com/drti/plaid/-/issues/new) and provide the following details:

- A clear and concise description of the feature.
- Any additional context or use cases.

## 4. Contributor License Agreement

In order to clarify the intellectual property license granted with Contributions from any person or entity, Safran must have a Contributor License Agreement ("CLA") on file that has been signed by each Contributor, indicating agreement to the license terms below. This license is for your protection as a Contributor as well as the protection of Safran; it does not change your rights to use your own Contributions for any other purpose.
You accept and agree to the following terms and conditions for Your present and future Contributions submitted to Safran. Except for the license granted herein to Safran and recipients of software distributed by Safran, You reserve all right, title, and interest in and to Your Contributions.

1. Definitions.
"You" (or "Your") shall mean the copyright owner or legal entity authorized by the copyright owner that is making this Agreement with Safran. For legal entities, the entity making a Contribution and all other entities that control, are controlled by, or are under common control with that entity are considered to be a single Contributor. For the purposes of this definition, "control" means (i) the power, direct or indirect, to cause the direction or management of such entity, whether by contract or otherwise, or (ii) ownership of fifty percent (50%) or more of the outstanding shares, or (iii) beneficial ownership of such entity.
"Contribution" shall mean any original work of authorship, including any modifications or additions to an existing work, that is intentionally submitted by You to Safran for inclusion in, or documentation of, any of the products owned or managed by Safran (the "Work"). For the purposes of this definition, "submitted" means any form of electronic, verbal, or written communication sent to Safran or its representatives, including but not limited to communication on electronic mailing lists, source code control systems, and issue tracking systems that are managed by, or on behalf of, Safran for the purpose of discussing and improving the Work, but excluding communication that is conspicuously marked or otherwise designated in writing by You as "Not a Contribution."

2. Grant of Copyright License. Subject to the terms and conditions of this Agreement, You hereby grant to Safran and to recipients of software distributed by Safran a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare derivative works of, publicly display, publicly perform, sublicense, and distribute Your Contributions and such derivative works.

3. Grant of Patent License. Subject to the terms and conditions of this Agreement, You hereby grant to Safran and to recipients of software distributed by Safran a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by You that are necessarily infringed by Your Contribution(s) alone or by combination of Your Contribution(s) with the Work to which such Contribution(s) was submitted. If any entity institutes patent litigation against You or any other entity (including a cross-claim or counterclaim in a lawsuit) alleging that your Contribution, or the Work to which you have contributed, constitutes direct or contributory patent infringement, then any patent licenses granted to that entity under this Agreement for that Contribution or Work shall terminate as of the date such litigation is filed.

4. You represent that you are legally entitled to grant the above license. If your employer(s) has rights to intellectual property that you create that includes your Contributions, you represent that you have received permission to make Contributions on behalf of that employer, that your employer has waived such rights for your Contributions to Safran, or that your employer has executed a separate Corporate CLA with Safran.

5. You represent that each of Your Contributions is Your original creation (see section 7 for submissions on behalf of others). You represent that Your Contribution submissions include complete details of any third-party license or other restriction (including, but not limited to, related patents and trademarks) of which you are personally aware and which are associated with any part of Your Contributions.

6. You are not expected to provide support for Your Contributions, except to the extent You desire to provide support. You may provide support for free, for a fee, or not at all. Unless required by applicable law or agreed to in writing, You provide Your Contributions on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON- INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE.

7. Should You wish to submit work that is not Your original creation, You may submit it to Safran separately from any Contribution, identifying the complete details of its source and of any license or other restriction (including, but not limited to, related patents, trademarks, and license agreements) of which you are personally aware, and conspicuously marking the work as "Submitted on behalf of a third-party: [named here]".

8. You agree to notify Safran of any facts or circumstances of which you become aware that would make these representations inaccurate in any respect.

---

Thank you for considering contributing to PLAID Project! Your help is greatly appreciated.
