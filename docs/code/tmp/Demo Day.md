

Summary

The interviewers will go very deep. They’ll switch from different techs quickly from high to low. Make sure you actively review the fundamentals, even if you think you already know them. During an interview, simple questions about epochs, batch size, tokens, PVCs, Pods, or networking can catch you off guard if they aren’t fresh in your mind. When you’re deep in Terraform, Soperator, multi-node training, and GPU architecture, the basics can slip into the background — and that’s exactly where candidates get tripped up.

WARNING: Do not assume your implicit knowledge will surface cleanly under pressure. Take time to refresh the core concepts so you can explain them quickly, clearly, and confidently. Strong recall of fundamentals prevents freezes, avoids rambling, and shows control. This is one of the most important parts of preparing for technical interviews. (Cheat Sheat Below)

**Model Selection Warning (Must Read Before Demo Day)**

**Do NOT pick a tiny model for your demo.**

Choose a **medium-sized model (at least 200M–1B parameters)** so you can clearly demonstrate:

1\. **Why you chose it** (task fit, architecture type, parameter size, performance-vs-cost tradeoff).

2\. **How you plan to get the best performance** (batch size tuning, distributed data parallelism, mixed precision, dataset size, and expected VRAM footprint).

3\. **What “good” looks like** (metrics to improve: ROUGE, BLEU, accuracy, perplexity, etc.).

Your interviewer wants to see that you understand **why this model**, **why this dataset**, **why this training strategy**, and **why this GPU type**.  
If the model is too small, you lose the opportunity to show that reasoning.

**Recommended sweet-spot choices:**

● **T5-base or T5-large** (text-to-text tasks)

● **BART-large** or **DistilBART-large variants**

● **Llama-3 8B (QLoRA fine-tune)**

● **GPT-J (6B) or GPT-NeoX (20B via LoRA)**

● **BERT-large** (for classification/embedding tasks)

These are big enough to require *real* GPU strategy, but small enough to train or fine-tune in a demo environment.

Questions asked

A. Terraform / Infrastructure Questions

**General reasoning**

1\. *“What did you change in the Terraform and why?”*

2\. *“Can you describe which decisions you made while modifying the Terraform?”*

3\. *“Why did you pick these specific settings?”*

4\. *“Could you have chosen a different configuration? What would performance differences be?”*

5\. *“Which Terraform settings influenced node distribution?”*

6\. *“How did you decide to split the 16 GPUs across two nodes?”*

**Storage choices**

7\. *“How did you create the storage volumes? Which settings did you use?”*

8\. *“What does block size mean?”*

9\. *“Which Nebius storage types exist and when would you use each?”*

10\. *“Does storage choice change if you train on 2 nodes or 200 nodes?”*

**Node pools / scaling**

11\. *“Why pick slurm-worker with size=2 instead of 4?”*

12\. *“Which configuration would give best performance?”*

B. Kubernetes Fundamentals Questions

**Basic constructs**

13\. *“What is a namespace?”*

14\. *“What is a Persistent Volume Claim (PVC)?”*

15\. *“How does a PVC relate to a PV?”*

16\. *“What is a StatefulSet? What’s a Deployment? A ReplicaSet?”*

17\. *“Differences between StatefulSet / Deployment / ReplicaSet?”*

**Scheduling**

18\. *“How do you force a pod to run on a specific node?”*

19\. *“Explain taints and tolerations vs labels and selectors.”*

**Storage interactions**

20\. *“Why was the PVC failing to bind?”*

21\. *“How do you debug a pod stuck pending due to PVC issues?”*

**GPU driver / runtime**

22\. *“How do you check which CUDA driver version is installed?”*

23\. *“How do you check GPU driver from inside a node?”*

C. Slurm / Soperator Questions

24\. *“Why did you upload the Python file twice?”*

25\. *“Why did you put the .py file under /tmp?”*

26\. *“Explain the \-N and namespace flags in kubectl cp.”*

27\. *“Why create a virtualenv inside your sbatch?”*

28\. *“Why install packages inside the sbatch instead of once in the jail?”*

29\. *“Would rerunning the job reinstall everything every time?”*

30\. *“Why not pre-install dependencies on the shared root filesystem?”*

D. Networking / GPU Fabric Questions

31\. *“What is InfiniBand?”*

32\. *“Difference between InfiniBand and NVLink?”*

E. Distributed Training Questions

33\. *“What is an epoch?”*

34\. *“What is a learning rate?”*

35\. *“What happens if LR is too high or too low?”*

36\. *“What is batch size?”*

37\. *“What issues occur if batch size is too big or too small?”*

38\. *“How many parameters did your model have?”*

39\. *“Given 80GB VRAM on H100, what’s the largest model you can fit?”*

Performance debugging

40\. *“Did you check GPU utilization?”*

41\. *“Why might your training only use one node?”*

42\. *“What would cause GPU utilization to be stuck at 50% across all nodes?”*

43\. *“If training is slower on Nebius than Google Cloud, what could be the reason?”*

44\. *“How would you debug multi-node slowness?”*

F. Model Architecture / NLP Basics

45\. *“Why did you choose DistilBART?”*

46\. *“Why a distilled model instead of a full model?”*

47\. *“What does seq-to-seq mean?”*

48\. *“Difference between encoder-only / decoder-only / encoder-decoder models?”*

49\. *“Which tasks fit encoder models? Decoder models? Seq2seq models?”*

50\. *“What is a token?”*

51\. *“Why is tokenization important?”*

52\. *“What is an embedding?”*

G. Inference Optimization Questions

53\. *“If inference is slow (10 seconds to first token), how would you speed it up?”*

54\. *“What model-level changes could speed up inference?”*

H. Misc Interview / Experience Questions

55\. *“What’s the largest cluster you’ve trained on?”*

56\. *“How much time did you dedicate to the project?”*

57\. *“What were the toughest mistakes you ran into?”*

**Fundamentals Quick Reference**

*(Memorize these. Say them clean and fast.)*

**ML Training Basics**

**Epoch**

One full pass through the training dataset.

**Batch Size**

Number of samples processed before one gradient update. Too big → GPU OOM.  
Too small → Underutilized GPUs, slow training.

**Gradient Accumulation**

Simulates a larger batch size when GPU memory is limited.

**Learning Rate**

How big each optimization step is. Too high → overshooting, unstable.  
Too low → training is slow, may get stuck.

**Loss Function**

A numeric measure of how wrong the model is.

**Token**

Smallest text unit the model processes (word, subword, symbol).

**Embedding**

A vector representation of a token’s meaning.

**Seq2Seq**

Encoder → Decoder architecture (translation, summarization).

**Encoder-only**

BERT-style (classification, embedding tasks).

**Decoder-only**

GPT-style (text generation).

**GPU & Distributed Training**

**GPU VRAM Rule of Thumb**

Model parameters × 2–4 bytes × \~3 for training overhead.

**NVLink**

High-speed GPU-to-GPU connection *inside* the same server.

**InfiniBand**

High-speed, low-latency *cluster* interconnect between servers.

**NCCL**

NVIDIA library that synchronizes gradients across GPUs.

**DDP (Distributed Data Parallel)**

Each GPU gets a shard of data; gradients are synced each step.

**Low GPU Utilization Causes**

● Batch size too small

● Slow data loading

● Network bottleneck (Ethernet vs IB)

● Model too small for distributed training

● Wrong env variables (DDP misconfig)

**Kubernetes Fundamentals**

**Pod**

Smallest deployable unit; runs one or more containers.

**Node**

A machine (VM) where Pods run.

**Deployment**

Manages stateless Pods; handles rollout & replicas.

**ReplicaSet**

Ensures the desired number of Pod copies exist (managed by Deployment).

**StatefulSet**

For stateful workloads — stable identity \+ persistent storage.

**Namespace**

Logical grouping/partition inside a Kubernetes cluster.

**PVC (PersistentVolumeClaim)** A request for storage by a Pod.

**PV (PersistentVolume)**

The actual storage resource.

**NodeSelector / Taints & Tolerations** Controls which Pods run on which nodes.

**Storage Basics**

**Object Storage (S3-compatible)**

Unstructured blobs (text, images, video). Ideal for datasets.

**Block Storage**

Attached disks; low latency; used for VMs, databases.

**File Storage (NFS / Filestore)**

Shared POSIX filesystem; required for Soperator’s Jail FS.

**Jail FS**

A shared root filesystem mounted across Slurm worker/login nodes.

**Slurm Basics**

**sbatch**

Submit a batch job.

**srun**

Run a command interactively or allocate resources.

**sinfo**

View node/partition status.

**squeue**

List running/pending jobs.

**Partition**

A queue with specific hardware and scheduling rules.

**Inference Basics**

**Ways to Speed Up Inference**

● fp16 / bf16

● Quantization (8-bit / 4-bit)

● TensorRT

● FlashAttention

● Increase batch size (for throughput)

● Use GPU with larger VRAM