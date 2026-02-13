# PIPr: A Dataset of Public Infrastructure as Code Programs

Programming Languages Infrastructure as Code (PL-IaC) enables IaC programs written in general-purpose programming languages like Python and TypeScript. The currently available PL-IaC solutions are Pulumi and the Cloud Development Kits (CDKs) of Amazon Web Services (AWS) and Terraform. This dataset provides metadata and initial analyses of all public GitHub repositories in August 2022 with an IaC program, including their programming languages, applied testing techniques, and licenses. Further, we provide a shallow copy of the head state of those 7104 repositories whose licenses permit redistribution. The dataset is available under the Open Data Commons Attribution License (ODC-By) v1.0.

Contents:

* `metadata.zip`: The dataset metadata and analysis results as CSV files.
* `scripts-and-logs.zip`: Scripts and logs of the dataset creation.
* `LICENSE`: The Open Data Commons Attribution License (ODC-By) v1.0 text.
* `README.md`: This document.
* `redistributable-repositiories.zip`: Shallow copies of the head state of all redistributable repositories with an IaC program.

This artifact is part of the ProTI Infrastructure as Code testing project: https://proti-iac.github.io.

## Metadata

The dataset's metadata comprises three tabular CSV files containing metadata about all analyzed repositories, IaC programs, and testing source code files.

`repositories.csv`:

* `ID` (integer): GitHub repository ID
* `url` (string): GitHub repository URL
* `downloaded` (boolean): Whether cloning the repository succeeded
* `name` (string): Repository name
* `description` (string): Repository description
* `licenses` (string, list of strings): Repository licenses
* `redistributable` (boolean): Whether the repository's licenses permit redistribution
* `created` (string, date & time): Time of the repository's creation
* `updated` (string, date & time): Time of the last update to the repository
* `pushed` (string, date & time): Time of the last push to the repository
* `fork` (boolean): Whether the repository is a fork
* `forks` (integer): Number of forks
* `archive` (boolean): Whether the repository is archived
* `programs` (string, list of strings): Project file path of each IaC program in the repository

`programs.csv`:

* `ID` (string): Project file path of the IaC program
* `repository` (integer): GitHub repository ID of the repository containing the IaC program
* `directory` (string): Path of the directory containing the IaC program's project file
* `solution` (string, enum): PL-IaC solution of the IaC program ("AWS CDK", "CDKTF", "Pulumi")
* `language` (string, enum): Programming language of the IaC program (enum values: "csharp", "go", "haskell", "java", "javascript", "python", "typescript", "yaml")
* `name` (string): IaC program name
* `description` (string): IaC program description
* `runtime` (string): Runtime string of the IaC program
* `testing` (string, list of enum): Testing techniques of the IaC program (enum values: "awscdk", "awscdk_assert", "awscdk_snapshot", "cdktf", "cdktf_snapshot", "cdktf_tf", "pulumi_crossguard", "pulumi_integration", "pulumi_unit", "pulumi_unit_mocking")
* `tests` (string, list of strings): File paths of IaC program's tests 

`testing-files.csv`:

* `file` (string): Testing file path
* `language` (string, enum): Programming language of the testing file (enum values: "csharp", "go", "java", "javascript", "python", "typescript")
* `techniques` (string, list of enum): Testing techniques used in the testing file (enum values: "awscdk", "awscdk_assert", "awscdk_snapshot", "cdktf", "cdktf_snapshot", "cdktf_tf", "pulumi_crossguard", "pulumi_integration", "pulumi_unit", "pulumi_unit_mocking")
* `keywords` (string, list of enum): Keywords found in the testing file (enum values: "/go/auto", "/testing/integration", "@AfterAll", "@BeforeAll", "@Test", "@aws-cdk", "@aws-cdk/assert", "@pulumi.runtime.test", "@pulumi/", "@pulumi/policy", "@pulumi/pulumi/automation", "Amazon.CDK", "Amazon.CDK.Assertions", "Assertions_", "HashiCorp.Cdktf", "IMocks", "Moq", "NUnit", "PolicyPack(", "ProgramTest", "Pulumi", "Pulumi.Automation", "PulumiTest", "ResourceValidationArgs", "ResourceValidationPolicy", "SnapshotTest()", "StackValidationPolicy", "Testing", "Testing_ToBeValidTerraform(", "ToBeValidTerraform(", "Verifier.Verify(", "WithMocks(", "[Fact]", "[TestClass]", "[TestFixture]", "[TestMethod]", "[Test]", "afterAll(", "assertions", "automation", "aws-cdk-lib", "aws-cdk-lib/assert", "aws_cdk", "aws_cdk.assertions", "awscdk", "beforeAll(", "cdktf", "com.pulumi", "def test_", "describe(", "github.com/aws/aws-cdk-go/awscdk", "github.com/hashicorp/terraform-cdk-go/cdktf", "github.com/pulumi/pulumi", "integration", "junit", "pulumi", "pulumi.runtime.setMocks(", "pulumi.runtime.set_mocks(", "pulumi_policy", "pytest", "setMocks(", "set_mocks(", "snapshot", "software.amazon.awscdk.assertions", "stretchr", "test(", "testing", "toBeValidTerraform(", "toMatchInlineSnapshot(", "toMatchSnapshot(", "to_be_valid_terraform(", "unittest", "withMocks(")
* `program` (string): Project file path of the testing file's IaC program

## Dataset Creation

`scripts-and-logs.zip` contains all scripts and logs of the creation of this dataset. In it, `executions/executions.log` documents the commands that generated this dataset in detail. On a high level, the dataset was created as follows:

1. A list of all repositories with a PL-IaC program configuration file was created using `search-repositories.py` (documented below). The execution took two weeks due to the non-deterministic nature of GitHub's REST API, causing excessive retries.
2. A shallow copy of the head of all repositories was downloaded using `download-repositories.py` (documented below).
3. Using `analysis.ipynb`, the repositories were analyzed for the programs' metadata, including the used programming languages and licenses.
4. Based on the analysis, all repositories with at least one IaC program and a redistributable license were packaged into `redistributable-repositiories.zip`, excluding any `node_modules` and `.git` directories.

## Searching Repositories

The repositories are searched through `search-repositories.py` and saved in a CSV file.
The script takes these arguments in the following order:

1. Github access token.
2. Name of the CSV output file.
3. Filename to search for.
4. File extensions to search for, separated by commas.
5. Min file size for the search (for all files: `0`).
6. Max file size for the search or `*` for unlimited (for all files: `*`).

Pulumi projects have a `Pulumi.yaml` or `Pulumi.yml` (case-sensitive file name) file in their root folder, i.e., (3) is `Pulumi` and (4) is `yml,yaml`. 
https://www.pulumi.com/docs/intro/concepts/project/

AWS CDK projects have a `cdk.json` (case-sensitive file name) file in their root folder, i.e., (3) is `cdk` and (4) is `json`.
https://docs.aws.amazon.com/cdk/v2/guide/cli.html

CDK for Terraform (CDKTF) projects have a `cdktf.json` (case-sensitive file name) file in their root folder, i.e., (3) is `cdktf` and (4) is `json`.
https://www.terraform.io/cdktf/create-and-deploy/project-setup

### Limitations

The script uses the GitHub code search API and inherits its limitations:

* Only forks with more stars than the parent repository are included.
* Only the repositories' default branches are considered.
* Only files smaller than 384 KB are searchable.
* Only repositories with fewer than 500,000 files are considered.
* Only repositories that have had activity or have been returned in search results in the last year are considered.

More details: https://docs.github.com/en/search-github/searching-on-github/searching-code

The results of the GitHub code search API are not stable.
However, the generally more robust GraphQL API does not support searching for files in repositories:
https://stackoverflow.com/questions/45382069/search-for-code-in-github-using-graphql-v4-api

## Downloading Repositories

`download-repositories.py` downloads all repositories in CSV files generated through `search-respositories.py` and generates an overview CSV file of the downloads.
The script takes these arguments in the following order:

1. Name of the repositories CSV files generated through `search-repositories.py`, separated by commas.
2. Output directory to download the repositories to.
3. Name of the CSV output file.

The script only downloads a shallow recursive copy of the HEAD of the repo, i.e., only the main branch's most recent state, including submodules, without the rest of the git history.
Each repository is downloaded to a subfolder named by the repository's ID.