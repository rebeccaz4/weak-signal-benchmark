# gpt-5.4 Epistemic AI Weak Signal Discovery Process

- Target topic: `epistemic AI`
- Model: `gpt-5.4`

## Upstream Topic Selection

These topics were selected because each retrieved more than 20,000 papers, making them broad enough to require a finer-grained topic layer before weak-signal construction.

The fine-grained topics were produced in two ways:

1. First, search for a direct 2024 survey for the broad topic, using queries such as `{topic} 2024 survey`.
2. If no suitable direct survey was available, generate many candidate fine-grained topics, search each candidate for survey coverage, keep the candidates with survey support, and then apply human review.

This list is not intended to be complete. The retained fine-grained topics are included because they passed the survey-support and human-review filter.

## Fine-Grained Topic Pairs

| Topic | Fine-grained topic |
|---|---|
| Artificial Intelligence of Things | AIoT sensing |
| Artificial Intelligence of Things | AIoT computing |
| Artificial Intelligence of Things | AIoT networking and communication |
| Artificial Intelligence of Things | AIoT domain-specific systems |
| explainable AI | XAI systems |
| explainable AI | XAI design methodology |
| explainable AI | explanation techniques |
| explainable AI | explainable models |
| explainable AI | application areas of explainable AI |
| explainable AI | evaluation of XAI techniques |
| explainable AI | challenges and future research directions in XAI |
| human AI interface | Human-Centered Explainable AI Interface |
| human AI interface | interactive AI |
| human AI interface | human-AI collaboration |
| human AI interface | generative-AI user interface patterns |
| Large Language Models | LLM Foundations |
| Large Language Models | LLM Adaptation |
| Large Language Models | LLM Augmentation |
| Large Language Models | LLM Agents |
| Large Language Models | LLM Reasoning |
| Large Language Models | LLM Evaluation |
| Large Language Models | LLM Reliability |
| scientific machine learning | neural operators and operator learning |
| scientific machine learning | Surrogate Modeling |
| scientific machine learning | Inverse Problems and Scientific Discovery |
| scientific machine learning | Differentiable Scientific Computing |
| scientific machine learning | hybrid scientific-model and data-driven learning |

## Prompt Used

### System Prompt

```text
You are an expert research topic extractor.
Your task is to extract literature-level research topics from paper abstracts.
Extract only topics that are explicitly discussed in the paper.
A good topic should be broad enough that multiple independent papers could study it,
but specific enough to be more informative than a general field label.
Return only valid json.
```

### User Prompt Template

```text
Target established topic:
{target_topic}

This paper was retrieved as related to the target established topic above.
Use the target topic only as context for relevance filtering.
Do not output the target topic itself unless the abstract discusses a more specific reusable subtopic.

Paper metadata:
- Title: {title}
- Paper ID: {paper_id}
- Year: {year}
- Venue: {venue}
- Source query: {source_query}

Abstract:
{abstract}

Topic categories:

1. Problem-space topics:
Research problems, gaps, limitations, risks, bottlenecks, evaluation failures, or scientific questions discussed by the paper.

2. Solution-space topics:
Research methods, method families, system directions, evaluation approaches, defenses, or solution directions discussed by the paper.
Use solution-space only for standalone reusable methods, method families, systems, defenses, algorithms, datasets, benchmarks, or evaluation protocols, not for the problem that motivates them.

Task:
Extract candidate research topics from this paper abstract that are conceptually related to the target established topic: "{target_topic}".
These candidates may be problem-space topics or solution-space topics.

Specificity guidance:
- Too broad: a whole field (e.g., "machine learning", "computer vision"), broad model family (e.g., "deep learning"), or generic category label (e.g., "optimization").
- Too specific: a paper-specific method name, system name, exact task setting, implementation detail, single experimental finding, or enumerating technical details.
- Correct level: a reusable research direction or problem space that multiple independent papers could study using different methods, or systems. The topic should capture the core research direction without enumerating specific techniques or data types.
- Focus on the research problem or method family, not on the specific implementation details or data modalities.
- If the abstract discusses a narrow technique or case study, abstract it to the broader research problem or method family it addresses.
- Do not phrase topics as actions or as this specific paper's contribution.
- A candidate topic must be a standalone research topic phrase, not a relation between a problem and a solution.
- Avoid "X for Y" topic names when X is a method and Y is a problem, goal, task, or desired property. Split them into separate candidate topics when both are explicitly supported.

Requirements:
- Output a json object with a "topics" array, which may be empty.
- Extract up to 2 topics total.
- Each topic must be explicitly supported by the abstract.
- Each topic must be reusable across multiple papers.
- Each topic must be one abstraction level broader than the paper's specific method, benchmark, dataset, or case study.
- Each topic must include exactly one "topic_type": "problem-space" or "solution-space". A mix is not allowed.
- Each topic must be conceptually related to the target established topic: "{target_topic}".
- If a topic cannot be clearly classified as problem-space or solution-space, do not emit it.
- If a topic cannot be clearly related to the target established topic, do not emit it.
- Do not phrase topics as actions, paper contributions, or problem-solution relationships.
- Do not combine a method and the problem it addresses into one topic phrase.
- Do not invent evidence beyond the abstract.

Return only json with this schema:
{{
  "topics": [
    {{
      "topic": "<standalone literature-level candidate topic>",
      "topic_type": "problem-space|solution-space",
      "target_topic": "{target_topic}",
      "evidence": "<short phrase grounded in the abstract>",
      "confidence": "high|medium|low"
    }}
  ]
}}
```

## Candidate Extraction Examples

## 1. Towards Empathetic Planning

- Paper ID: `56373ceeab7a580853a10d67f84b9aff5abc6a7d`
- Year: 2019
- Venue: arXiv.org

**Abstract**

Critical to successful human interaction is a capacity for empathy - the ability to understand and share the thoughts and feelings of another. As Artificial Intelligence (AI) systems are increasingly required to interact with humans in a myriad of settings, it is important to enable AI to wield empathy as a tool to benefit those it interacts with. In this paper, we work towards this goal by bringing together a number of important concepts: empathy, AI planning, and reasoning in the presence of knowledge and belief. We formalize the notion of Empathetic Planning which is informed by the beliefs and affective state of the empathizee. We appeal to an epistemic logic framework to represent the beliefs of the empathizee and propose AI planning-based computational approaches to compute empathetic solutions. We illustrate the potential benefits of our approach by conducting a study where we evaluate participants' perceptions of the agent's empathetic abilities and assistive capabilities.

**Extracted Topics**

- Topic: empathetic planning
  - Type: solution-space
  - EU Theme: epistemic AI
  - Confidence: high
  - Evidence: We formalize the notion of Empathetic Planning which is informed by the beliefs and affective state of the empathizee.
- Topic: epistemic planning
  - Type: solution-space
  - EU Theme: epistemic AI
  - Confidence: high
  - Evidence: We appeal to an epistemic logic framework to represent the beliefs of the empathizee and propose AI planning-based computational approaches.

## 2. In AI We Trust Incrementally: a Multi-layer Model of Trust to Analyze Human-Artificial Intelligence Interactions

- Paper ID: `c559fb094ba4f111d9134f0df9bef3a7b0a050db`
- Year: 2019
- Venue: Philosophy & Technology

**Abstract**

Real engines of the artificial intelligence (AI) revolution, machine learning (ML) models, and algorithms are embedded nowadays in many services and products around us. As a society, we argue it is now necessary to transition into a phronetic paradigm focused on the ethical dilemmas stemming from the conception and application of AIs to define actionable recommendations as well as normative solutions. However, both academic research and society-driven initiatives are still quite far from clearly defining a solid program of study and intervention. In this contribution, we will focus on selected ethical investigations around AI by proposing an incremental model of trust that can be applied to both human-human and human-AI interactions. Starting with a quick overview of the existing accounts of trust, with special attention to Taddeo’s concept of “e-trust,” we will discuss all the components of the proposed model and the reasons to trust in human-AI interactions in an example of relevance for business organizations. We end this contribution with an analysis of the epistemic and pragmatic reasons of trust in human-AI interactions and with a discussion of kinds of normativity in trustworthiness of AIs.

**Extracted Topics**

- Topic: trust in human-AI interactions
  - Type: problem-space
  - EU Theme: epistemic AI
  - Confidence: high
  - Evidence: we will focus on selected ethical investigations around AI by proposing an incremental model of trust ... in human-AI interactions
- Topic: incremental models of trust
  - Type: solution-space
  - EU Theme: epistemic AI
  - Confidence: high
  - Evidence: proposing an incremental model of trust that can be applied to both human-human and human-AI interactions

## Weak Signal Results

| Target topic | Candidate topic | Type | f_early | f_later | growth | impact | weak? |
|---|---|---|---:|---:|---:|---:|---|
| epistemic AI | epistemic and control gaps in AI responsibility attribution | problem-space | 0.000000 | 0.009756 | 0.009756 | 0.134785 | yes |
| epistemic AI | epistemic injustice in AI | problem-space | 0.000000 | 0.009756 | 0.009756 | 0.134785 | yes |
| epistemic AI | moral responsibility gaps in AI-supported decision-making | problem-space | 0.000000 | 0.009756 | 0.009756 | 0.134785 | yes |
| epistemic AI | AI impacts on professional epistemic authority | problem-space | 0.000000 | 0.004878 | 0.004878 | 0.067393 | yes |
| epistemic AI | epistemic authority in AI governance | problem-space | 0.000000 | 0.004878 | 0.004878 | 0.067393 | yes |
| epistemic AI | epistemic diversity in AI ethics research | problem-space | 0.000000 | 0.004878 | 0.004878 | 0.067393 | yes |
| epistemic AI | epistemic oppression in AI systems | problem-space | 0.000000 | 0.004878 | 0.004878 | 0.067393 | yes |
| epistemic AI | epistemic diversity in AI development | solution-space | 0.000000 | 0.004878 | 0.004878 | 0.067393 | yes |
| epistemic AI | epistemic insight in AI-mediated science learning | solution-space | 0.000000 | 0.004878 | 0.004878 | 0.067393 | yes |
| epistemic AI | epistemic program verification | problem-space | 0.004098 | 0.004878 | 0.000780 | 0.004286 | yes |
