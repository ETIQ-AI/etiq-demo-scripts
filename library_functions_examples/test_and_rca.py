"""
Example script showing how to use the etiq_copilot library 
- to inspect a python script (user or agent made)
- to capture all the artifacts (no manual instrumentation needed)
- to retrieve: 
    - tests
    - rca

Install the etiq_copilot library; test release:
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "etiq-copilot==2.3.0-rc3"
uv pip install --index-strategy unsafe-best-match --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "etiq-copilot==2.3.0-rc3"

Steps shown in this script:
1. Run the etiq copilot scanner on a target Python file in this case `iris_lineage_test.py`.
2. Example of how to generate test and results.
3. Example of how to retrieve root cause analysis of the issue in case test fails.

To try it on your own script point it to the file that initializes the run.
For limitations, see the docs.
"""


from  pathlib import Path
import logging


#from etiq_copilot.engine.daemons.utils import working_directory
from etiq_copilot.engine.implementations.scanner.code_scanner import DebuggerCodeScanner
from etiq_copilot.engine.implementations.scanner.scan_results import CodeScannerResult
#from verification_functions_codex import get_empty_objects
#from etiq_copilot.engine.daemons.runner_utils import run_rca_on_test_results, run_tests

from collections import defaultdict

from etiq_copilot.engine.daemons.utils import (
    default_rca_recommender_repo_factory,
    default_test_recommender_repo_factory,
)
from etiq_copilot.engine.entities.rca_config import EtiqTestContext
from etiq_copilot.engine.entities.results import EntityTestResult, RCAResults
from etiq_copilot.engine.entities.scanner.base_state_objects import State
from etiq_copilot.engine.implementations.rca import EtiqLineageRCARunner
from etiq_copilot.engine.implementations.runners import (
    EtiqV1ContextAwareRunner,
)
from etiq_copilot.engine.implementations.test_recommenders import (
    RecommenderRepository,
)
from etiq_copilot.engine.interfaces.rca import (
    AbstractRCARecommender,
)
from etiq_copilot.engine.interfaces.scanner import (
    AbstractCodeScannerResult,
)
from etiq_copilot.engine.telemetry import BaseTelemetry



#Assign your file path 

your_file_path = r"library_functions/iris_lineage_test.py"
logger = logging.getLogger(__name__)

def scan_file(
    scan_file_path: Path | str,
) -> CodeScannerResult:
    """Analyze Python code and return scan results.

    This function serves as the main entry point for the code analysis tool.
    """
    scan_file_path = Path(scan_file_path)
    scan_results = CodeScannerResult()
    original_code: str | None = None
    test_scanner = DebuggerCodeScanner()
    try:
        original_code = Path(scan_file_path).read_text(encoding="utf-8")
        #with (
        #    working_directory(scan_file_path.parent),
        #):
        scan_results = test_scanner.scan_code(code_str=original_code)
    except Exception:
        raise
    return scan_results


def run_tests(
    scan_results: AbstractCodeScannerResult,
    test_recommender_repository: RecommenderRepository | None = None,
    test_runner: EtiqV1ContextAwareRunner | None = None,
    rca_repo: AbstractRCARecommender | None = None,
    max_tests: int = 15,
    telemetry: BaseTelemetry | None = None,
) -> dict[State, list[EntityTestResult]]:
    """Run recommended tests against scan results and return their outcomes.

    Executes test recommendations generated from code scan results, using
    a context-aware test runner. Initializes default repositories if not
    provided. Runs up to max_tests recommendations per candidate test state.

    Args:
        scan_results: The code scanner results containing candidate test states
            to run tests against.
        test_recommender_repository: Repository for generating test recommendations.
            Defaults to the default test recommender repository if None.
        test_runner: The runner instance for executing tests. Will be created
            internally if None is provided.
        rca_repo: Root cause analysis recommender repository. Defaults to the
            default RCA recommender repository if None.
        max_tests: Maximum number of test recommendations to run per state.
            Defaults to 15.
        telemetry: Optional telemetry instance for tracking test execution metrics.

    Returns:
        A list of EntityTestResult objects containing the outcomes of all
        executed tests. Silently skips tests that raise exceptions.

    """
    if test_recommender_repository is None:
        test_recommender_repository = default_test_recommender_repo_factory()
    if rca_repo is None:
        rca_repo = default_rca_recommender_repo_factory()
    if test_runner is None:
        test_runner = EtiqV1ContextAwareRunner(
            code_scan_results=scan_results,
            rca_recommender=rca_repo,
            telemetry=telemetry,
        )
    test_results: dict[State, list[EntityTestResult]] = defaultdict(list)
    for state in scan_results.debug_candidate_test_states:
        # Run Tests for at most "max_tests" test recommendations
        for rec in test_recommender_repository.recommend(state)[:max_tests]:
            try:
                test_result = test_runner.run(rec)
                test_results[state].append(test_result)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "Unhandled exception while running %s on variable with names %s",
                    rec.name,
                    state.names,
                )
    return test_results



def run_rca_on_test_results(
    test_results_dict: dict[State, list[EntityTestResult]],
    scan_results: AbstractCodeScannerResult,
) -> dict[State, tuple[EntityTestResult, RCAResults]]:
    """Run root cause analysis on test results that have associated RCA configurations.

    This function processes a list of test results and performs RCA analysis on those
    that:
    - Have an RCA configuration defined
    - Contain at least one issue
    - Have an associated test

    Args:
        test_results_dict: A dictionary of EntityTestResult objects (indexed by State)
            to analyze.
        scan_results: An AbstractCodeScannerResult containing the code scanning
            information used to build the test context for RCA analysis.

    Returns:
        A dictionary mapping EntityTestResult objects to their corresponding
        RCAResults, containing the root cause analysis findings for each test result
        that underwent analysis.

    """
    issues_mappings: dict[State, tuple[EntityTestResult, RCAResults]] = {}
    for state, results in test_results_dict.items():
        for res in results:
            if (
                res.rca_config is not None
                and res.number_of_issues() > 0
                and res.test is not None
            ):
                initial_test_context = EtiqTestContext.from_test(
                    test_config=res.test,
                    scan_results=scan_results,
                )
                issues_mappings[state] = (
                    res,
                    EtiqLineageRCARunner.run(
                        rca_config=res.rca_config,
                        test_context=initial_test_context,
                        code_scanner_result=scan_results,
                    ),
                )
    return dict(
        sorted(
            [(k, v) for (k, v) in issues_mappings.items()],
            key=lambda x: x[0].line_no,
        ),
    )


scan_results = scan_file(scan_file_path=your_file_path)

test_results = run_tests(scan_results)
rca_results = run_rca_on_test_results(scan_results = scan_results, test_results_dict = test_results)

print(test_results)

print(rca_results)

