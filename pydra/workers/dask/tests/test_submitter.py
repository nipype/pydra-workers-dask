import subprocess as sp
import time
import pytest
from pydra.engine.submitter import Submitter
from pydra.compose import python, workflow
from .utils import BasicWorkflow
import logging

logger = logging.getLogger("pydra.worker")


@python.define
def SleepAddOne(x):
    time.sleep(1)
    return x + 1


def test_callable_wf(any_worker, tmpdir):
    wf = BasicWorkflow(x=5)
    outputs = wf(cache_dir=tmpdir)
    assert outputs.out == 9
    del wf, outputs

    # providing any_worker
    wf = BasicWorkflow(x=5)
    outputs = wf(worker="cf")
    assert outputs.out == 9
    del wf, outputs

    # providing plugin_kwargs
    wf = BasicWorkflow(x=5)
    outputs = wf(worker="cf", n_procs=2)
    assert outputs.out == 9
    del wf, outputs

    # providing wrong plugin_kwargs
    wf = BasicWorkflow(x=5)
    with pytest.raises(TypeError, match="an unexpected keyword argument"):
        wf(worker="cf", sbatch_args="-N2")

    # providing submitter
    wf = BasicWorkflow(x=5)

    with Submitter(worker=any_worker, cache_dir=tmpdir) as sub:
        res = sub(wf)
    assert res.outputs.out == 9


def test_concurrent_wf(any_worker, tmpdir):
    # concurrent workflow
    # A --> C
    # B --> D
    @workflow.define(outputs=["out1", "out2"])
    def Workflow(x, y):
        taska = workflow.add(SleepAddOne(x=x), name="taska")
        taskb = workflow.add(SleepAddOne(x=y), name="taskb")
        taskc = workflow.add(SleepAddOne(x=taska.out), name="taskc")
        taskd = workflow.add(SleepAddOne(x=taskb.out), name="taskd")
        return taskc.out, taskd.out

    wf = Workflow(x=5, y=10)

    with Submitter(worker=any_worker, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, " ".join(results.errors["error message"])
    outputs = results.outputs
    assert outputs.out1 == 7
    assert outputs.out2 == 12


def test_concurrent_wf_nprocs(tmpdir):
    # concurrent workflow
    # setting n_procs in Submitter that is passed to the worker
    # A --> C
    # B --> D
    @workflow.define(outputs=["out1", "out2"])
    def Workflow(x, y):
        taska = workflow.add(SleepAddOne(x=x), name="taska")
        taskb = workflow.add(SleepAddOne(x=y), name="taskb")
        taskc = workflow.add(SleepAddOne(x=taska.out), name="taskc")
        taskd = workflow.add(SleepAddOne(x=taskb.out), name="taskd")
        return taskc.out, taskd.out

    wf = Workflow(x=5, y=10)
    with Submitter(worker="cf", n_procs=2, cache_dir=tmpdir) as sub:
        res = sub(wf)

    assert not res.errored, " ".join(res.errors["error message"])
    outputs = res.outputs
    assert outputs.out1 == 7
    assert outputs.out2 == 12


def test_wf_in_wf(any_worker, tmpdir):
    """WF(A --> SUBWF(A --> B) --> B)"""

    # workflow task
    @workflow.define
    def SubWf(x):
        sub_a = workflow.add(SleepAddOne(x=x), name="sub_a")
        sub_b = workflow.add(SleepAddOne(x=sub_a.out), name="sub_b")
        return sub_b.out

    @workflow.define
    def WfInWf(x):
        a = workflow.add(SleepAddOne(x=x), name="a")
        subwf = workflow.add(SubWf(x=a.out), name="subwf")
        b = workflow.add(SleepAddOne(x=subwf.out), name="b")
        return b.out

    wf = WfInWf(x=3)

    with Submitter(worker=any_worker, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, " ".join(results.errors["error message"])
    outputs = results.outputs
    assert outputs.out == 7


@pytest.mark.flaky(reruns=2)  # when dask
def test_wf2(any_worker, tmpdir):
    """workflow as a node
    workflow-node with one task and no splitter
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(SleepAddOne(x=x))
        return add2.out

    @workflow.define
    def Workflow(x):
        wfnd = workflow.add(Wfnd(x=x))
        return wfnd.out

    wf = Workflow(x=2)

    with Submitter(worker=any_worker, cache_dir=tmpdir) as sub:
        res = sub(wf)

    assert res.outputs.out == 3


@pytest.mark.flaky(reruns=2)  # when dask
def test_wf_with_state(any_worker, tmpdir):
    @workflow.define
    def Workflow(x):
        taska = workflow.add(SleepAddOne(x=x), name="taska")
        taskb = workflow.add(SleepAddOne(x=taska.out), name="taskb")
        return taskb.out

    wf = Workflow().split(x=[1, 2, 3])

    with Submitter(cache_dir=tmpdir, worker=any_worker) as sub:
        res = sub(wf)

    assert res.outputs.out[0] == 3
    assert res.outputs.out[1] == 4
    assert res.outputs.out[2] == 5


def test_debug_wf():
    # Use serial any_worker to execute workflow instead of CF
    wf = BasicWorkflow(x=5)
    outputs = wf(worker="debug")
    assert outputs.out == 9


@python.define
def Sleep(x, job_name_part):
    time.sleep(x)

    # getting the job_id of the first job that sleeps
    job_id = 999
    while job_id != "":
        time.sleep(3)
        id_p1 = sp.Popen(["squeue"], stdout=sp.PIPE)
        id_p2 = sp.Popen(["grep", job_name_part], stdin=id_p1.stdout, stdout=sp.PIPE)
        id_p3 = sp.Popen(["awk", "{print $1}"], stdin=id_p2.stdout, stdout=sp.PIPE)
        job_id = id_p3.communicate()[0].decode("utf-8").strip()

    return x


@python.define
def Cancel(job_name_part):

    # getting the job_id of the first job that sleeps
    job_id = ""
    while job_id == "":
        time.sleep(1)
        id_p1 = sp.Popen(["squeue"], stdout=sp.PIPE)
        id_p2 = sp.Popen(["grep", job_name_part], stdin=id_p1.stdout, stdout=sp.PIPE)
        id_p3 = sp.Popen(["awk", "{print $1}"], stdin=id_p2.stdout, stdout=sp.PIPE)
        job_id = id_p3.communicate()[0].decode("utf-8").strip()

    # # canceling the job
    proc = sp.run(["scancel", job_id, "--verbose"], stdout=sp.PIPE, stderr=sp.PIPE)
    # cancelling the job returns message in the sterr
    return proc.stderr.decode("utf-8").strip()


def qacct_output_to_dict(qacct_output):
    stdout_dict = {}
    for line in qacct_output.splitlines():
        key_value = line.split(None, 1)
        if key_value[0] not in stdout_dict:
            stdout_dict[key_value[0]] = []
        if len(key_value) > 1:
            stdout_dict[key_value[0]].append(key_value[1])
        else:
            stdout_dict[key_value[0]].append(None)

    print(stdout_dict)
    return stdout_dict
