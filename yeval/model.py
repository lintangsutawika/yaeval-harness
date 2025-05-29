import os
import time
import subprocess

from yeval.utils import check_api_health

def get_host_and_port(api_base):
    if "/v1/" in api_base:
        api_base = api_base.split("/v1/")[0]
    elif api_base.endswith("/"):
        api_base = api_base[:-1]
    *host, port = api_base.split(":")
    host = ":".join(host)
    return host, port

class Server:
    def __init__(
        self,
        model_name,
        host="http://127.0.0.1",
        port=8000,
        backend="vllm",
        max_model_len=4096,
        pp_size=1, tp_size=1,
        load_balancing=False,
        lb_backend="litellm",
        lb_port=4000,
        num_instance=1,
        tmpdir="/tmp/",
    ):
        self.model_name = model_name
        self.host = host
        self.port = port
        self.backend = backend
        self.max_model_len = max_model_len
        self.pp_size = pp_size
        self.tp_size = tp_size
        self.load_balancing = load_balancing
        self.lb_backend = lb_backend
        self.lb_port = lb_port
        self.num_instance = num_instance
        self.tmpdir = tmpdir

    def start(self):

        if self.backend == "ollama":
            command = [
                "ollama",
                "run",
                self.model_name,
            ]
        elif self.backend == "vllm":
            command = [
                "vllm", "serve", self.model_name,
                "--max_model_len", str(self.max_model_len),
                "--pipeline_parallel_size", str(self.pp_size),
                "--tensor_parallel_size", str(self.tp_size),
                # "--max-num-seqs", str(32),
                # "--enable-chunked-prefill",
                # "--max_num_batched_tokens", str(8*self.max_model_len),
                # "--distributed-executor-backend", "mp"
            ]

        if self.load_balancing:
            self.process = []
            # Make config file
            config_path = os.path.join(self.tmpdir, "config.yaml")
            if os.path.exists(config_path):
                os.remove(config_path)

            with open(config_path, "w") as config_file:
                config_file.write("model_list:\n")

            for i in range(self.num_instance):
                port = self.port + i
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = f"{i}"
                process = subprocess.Popen(
                    command+["--port", str(port)], env=env,
                    shell=False, stdout=subprocess.DEVNULL
                    )
                self.process.append(process)
                print(f"{self.backend} server {self.model_name}, Port: {port}, started with PID: {process.pid}")
                with open(config_path, "a") as config_file:
                    config_file.write(f"  - model_name: {self.model_name}\n")
                    config_file.write(f"    litellm_params:\n")
                    config_file.write(f"      model: hosted_vllm/{self.model_name}\n")
                    config_file.write(f"      api_base: http://localhost:{port}/v1\n")
                    config_file.write(f"      api_key: EMPTY\n")
                    # config_file.write(f"      timeout: 30\n")
                    # config_file.write(f"      rpm: 1000\n")

            # with open(config_path, "a") as config_file:
            #     config_file.write(f"router_settings:\n")
            #     config_file.write(f"    routing_strategy: usage-based-routing-v2\n")

            while True:
                num_api_up = 0
                for i in range(self.num_instance):
                    port = self.port + i
                    if check_api_health(f"http://localhost:{self.port}/health/"):
                        num_api_up += 1
                
                if num_api_up == self.num_instance:
                    break
                print(f"Waiting for {self.num_instance-num_api_up} {self.backend} server to start...")
                time.sleep(15)

            print("LiteLLM")
            process = subprocess.Popen(
                ["litellm", "--config", config_path],
                shell=False, stdout=subprocess.DEVNULL,
            )
            self.process.append(process)
            print(f"LiteLLM server started with PID: {process.pid}")
        else:
            self.process = subprocess.Popen(
                command+["--port", str(self.port)],
                shell=False, stdout=subprocess.DEVNULL
                )
            print(f"{self.backend} server {self.model_name}, started with PID: {self.process.pid}")

            while True:
                if check_api_health(f"http://localhost:{self.port}/health/"):
                    break

                print(f"Waiting for {self.backend} server to start...")
                time.sleep(15)

        return self.process

    def stop(self, process=None):
        if process is None:
            process = self.process

        if self.backend == "ollama":
            command = ["ollama", "stop", self.model_name]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process.wait()
        elif self.backend == "vllm":
            if isinstance(process, list):
                for p in process:
                    p.terminate()
                    p.wait()
            else:
                process.terminate()
                process.wait()
        print(f"{self.backend} server terminated.")
