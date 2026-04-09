# AI Incident Response Environment (OpenEnv)

A robust, OpenEnv-compliant simulation environment designed for the training and evaluation of AI-driven Site Reliability Engineering (SRE) agents. This project simulates a microservices architecture experiencing various catastrophic failures, requiring an AI agent to inspect telemetry, diagnose root causes, and apply targeted remediations.

Built for the Meta PyTorch Hackathon x Scaler School of Technology.

**Author:** Asifa Bandulal Beedi

---

## Features

* **OpenEnv-Compliant API:** Fully exposes standard reinforcement learning endpoints (`/reset`, `/step`, `/state`) via a scalable FastAPI server.
* **Microservices Simulation:** Simulates a four-tier distributed architecture comprising an API gateway, authentication service, payments service, and database proxy.
* **Dynamic Signal-to-Noise Logging:** Generates real-time, interleaved log streams containing both routine system noise and critical diagnostic fault signals.
* **Automated Evaluation Resilience:** Incorporates comprehensive server-side exception handling to gracefully manage invalid JSON payloads, out-of-bounds state transitions, and edge-case inputs from automated test runners, preventing HTTP 500 internal server errors.
* **Bounded Reward System:** Implements a mathematically bounded evaluation function that guarantees cumulative episodic rewards remain strictly within the required interval of (0, 1), ensuring compliance with rigid automated validation pipelines.

---

## Available Tasks

This environment supports three distinct incident scenarios, graded by difficulty:

1. **auth-crash-loop (Easy)**
   * **Description:** The authentication service is caught in a continuous crash loop due to a missing environment variable. 
   * **Required Action:** Restart the authentication service.
   
2. **payments-oom-cascade (Medium)**
   * **Description:** The payments service encounters an Out-Of-Memory (OOM) termination, causing cascading connection timeouts upstream at the API gateway.
   * **Required Action:** Restart the payments service to resolve the root cause, with optional partial credit for mitigating symptoms at the gateway.
   
3. **network-split-db-leak (Hard)**
   * **Description:** Database Write-Ahead-Log (WAL) saturation causes severe latency spikes across three frontend services, generating conflicting diagnostic telemetry.
   * **Required Action:** Clear the database proxy cache/WAL to resolve the saturation, requiring careful management of rolling restarts to clear delayed system symptoms.

---

## Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/AsifaBeedi/incident-env-.git](https://github.com/AsifaBeedi/incident-env-.git)
cd incident-env-
