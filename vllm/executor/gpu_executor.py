from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import ExecuteModelRequest, PoolerOutput
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async)
from vllm.worker.worker_base import WorkerBase, WorkerWrapperBase

logger = init_logger(__name__)


def create_worker(worker_module_name: str, worker_class_name: str,
                  worker_class_fn: Optional[Callable[[], Type[WorkerBase]]],
                  **kwargs):
    wrapper = WorkerWrapperBase(
        worker_module_name=worker_module_name,
        worker_class_name=worker_class_name,
        worker_class_fn=worker_class_fn,
    )
    wrapper.init_worker(**kwargs)
    return wrapper.worker


class GPUExecutor(ExecutorBase):

    uses_ray: bool = False

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """
        assert self.parallel_config.world_size == 1, (
            "GPUExecutor only supports single GPU.")

        self.driver_worker = self._create_worker()
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def _get_worker_kwargs(
            self,
            local_rank: int = 0,
            rank: int = 0,
            distributed_init_method: Optional[str] = None) -> Dict[str, Any]:
        """Return worker init args for a given rank."""
        if distributed_init_method is None:
            distributed_init_method = get_distributed_init_method(
                get_ip(), get_open_port())
        return dict(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            load_config=self.load_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            speculative_config=self.speculative_config,
            prompt_adapter_config=self.prompt_adapter_config,
            is_driver_worker=(not self.parallel_config)
            or (rank % self.parallel_config.tensor_parallel_size == 0),
            observability_config=self.observability_config,
        )

    def _get_worker_module_and_class(
            self) -> Tuple[str, str, Optional[Callable[[], Type[WorkerBase]]]]:
        worker_class_fn = None
        if self.scheduler_config.is_multi_step:
            worker_module_name = "vllm.worker.multi_step_worker"
            worker_class_name = "MultiStepWorker"
        elif self.speculative_config:
            worker_module_name = "vllm.spec_decode.spec_decode_worker"
            worker_class_name = "create_spec_worker"
        else:
            worker_module_name = "vllm.worker.worker"
            worker_class_name = "Worker"
        return (worker_module_name, worker_class_name, worker_class_fn)

    def _get_create_worker_kwargs(
            self,
            local_rank: int = 0,
            rank: int = 0,
            distributed_init_method: Optional[str] = None) -> Dict:
        worker_kwargs = self._get_worker_kwargs(local_rank, rank,
                                                distributed_init_method)

        (worker_module_name, worker_class_name,
         worker_class_fn) = self._get_worker_module_and_class()
        worker_kwargs.update(
            worker_module_name=worker_module_name,
            worker_class_name=worker_class_name,
            worker_class_fn=worker_class_fn,
        )

        return worker_kwargs

    def _create_worker(self,
                       local_rank: int = 0,
                       rank: int = 0,
                       distributed_init_method: Optional[str] = None):
        return create_worker(**self._get_create_worker_kwargs(
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method))

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        print("GPUExecutor self.driver_worker", self.driver_worker)
        return self.driver_worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        # NOTE: This is logged in the executor because there can be >1 worker
        # with other executors. We could log in the engine level, but work
        # remains to abstract away the device for non-GPU configurations.
        logger.info("# GPU blocks: %d, # CPU blocks: %d", num_gpu_blocks,
                    num_cpu_blocks)

        self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def execute_model(
        self, execute_model_req: ExecuteModelRequest
    ) -> Optional[List[Union[SamplerOutput, PoolerOutput]]]:
        
        # GPUExecutor output [SamplerOutput(outputs=[CompletionSequenceGroupOutput(samples=[SequenceOutput(parent_seq_id=0, output_token=24, logprobs={24: Logprob(logprob=inf, rank=None, decoded_token=None)})], prompt_logprobs=None)], sampled_token_probs=None, sampled_token_ids=None, spec_decode_worker_metrics=None)]
        # GPUExecutor driver_worker <vllm.worker.worker.Worker object at 0x7f1265104e20>
        # GPUExecutor execute_model_req ExecuteModelRequest(seq_group_metadata_list=[SequenceGroupMetadata(request_id='chat-2054455c0c964d15baceb49a0bf8fe3b', is_prompt=False, seq_data={0: SequenceData(prompt_token_ids=array('l', [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 151645, 198, 151644, 872, 198, 102261, 102261, 99565, 99593, 101602, 18830, 24, 24, 18538, 3837, 102336, 24, 15, 18538, 3837, 99517, 97706, 100124, 100430, 18538, 80443, 50930, 11319, 151645, 198, 151644, 77091, 198]), output_token_ids=(102261, 102261, 97706, 100124, 24), cumulative_logprob=inf, get_num_computed_tokens=45}, sampling_params=SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.7, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=[], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=87, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), block_tables={0: [119075, 119074, 119073]}, do_sample=True, pooling_params=None, lora_request=None, computed_block_nums=[], state=SequenceGroupState(num_steps=1, current_step=0), multi_modal_data=None, encoder_seq_data=None, cross_block_table=None, prompt_adapter_request=None, token_chunk_size=1, num_speculative_tokens=None)], blocks_to_swap_in=[], blocks_to_swap_out=[], blocks_to_copy=[], virtual_engine=0, num_lookahead_slots=0, running_queue_size=1, previous_hidden_states=None, num_steps=1, finished_requests_ids=[], last_sampled_token_ids=None, async_callback=functools.partial(<function weak_bind.<locals>.weak_bound at 0x7f1261e80940>, ctx=<vllm.engine.llm_engine.SchedulerContext object at 0x7f1264f00100>))
                                                        
        print("GPUExecutor driver_worker", self.driver_worker)
        print("GPUExecutor execute_model_req", execute_model_req)
        output = self.driver_worker.execute_model(execute_model_req)
        print("GPUExecutor output", output)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return self.driver_worker.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self.driver_worker.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self.driver_worker.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.driver_worker.list_loras()

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        assert prompt_adapter_request.prompt_adapter_id > 0, \
            "prompt_adapter_id must be greater than 0."
        return self.driver_worker.add_prompt_adapter(prompt_adapter_request)

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        assert prompt_adapter_id > 0, \
            "prompt_adapter_id must be greater than 0."
        return self.driver_worker.remove_prompt_adapter(prompt_adapter_id)

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        assert prompt_adapter_id > 0, \
                "prompt_adapter_id must be greater than 0."
        return self.driver_worker.pin_prompt_adapter(prompt_adapter_id)

    def list_prompt_adapters(self) -> Set[int]:
        return self.driver_worker.list_prompt_adapters()

    def check_health(self) -> None:
        # GPUExecutor will always be healthy as long as
        # it's running.
        return

    def start_profile(self) -> None:
        self.driver_worker.start_profile()

    def stop_profile(self) -> None:
        self.driver_worker.stop_profile()


class GPUExecutorAsync(GPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[Union[SamplerOutput, PoolerOutput]]:
        output = await make_async(self.driver_worker.execute_model
                                  )(execute_model_req=execute_model_req)
        return output
