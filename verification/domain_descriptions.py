#!/usr/bin/env python
# coding: utf-8
"""
Domain names and descriptions used as Semantic Scholar search queries.

Each key matches a domain in ``construction/weak_signals_by_domain.json``.
The description is appended to the domain name when querying S2 so that the
returned papers are topically relevant to the domain.

Descriptions are sourced / adapted from Wikipedia.
"""
from __future__ import annotations

import re

DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "Advanced materials and advanced manufacturing": (
        "Materials science is an interdisciplinary field concerned with "
        "understanding the relationships between the structure of materials "
        "at atomic or molecular scales and their macroscopic properties, and "
        "using this knowledge to design materials for specific applications. "
        "Practitioners follow the processing-structure-properties-performance "
        "paradigm, where manufacturing methods determine structural outcomes "
        "which in turn govern material properties that ultimately control "
        "performance in real-world applications. Advanced manufacturing "
        "extends this by developing innovative fabrication processes such as "
        "additive manufacturing, laser processing, and precision engineering "
        "to translate novel materials into functional devices and components."
    ),
    "Aerospace": (
        "Aerospace is the human effort in science, engineering, and business "
        "to fly in the atmosphere of Earth and surrounding space. Aerospace "
        "organizations research, design, manufacture, operate, and maintain "
        "aircraft and spacecraft. The field encompasses both aeronautics, "
        "dealing with aircraft that operate within Earth's atmosphere, and "
        "astronautics, dealing with spacecraft that operate outside the "
        "atmosphere. Aerospace activity is very diverse, with a multitude of "
        "commercial, industrial, and military applications."
    ),
    "Artificial intelligence & Machine learning": (
        "Artificial intelligence (AI) is the capability of computational "
        "systems to perform tasks typically associated with human "
        "intelligence, such as learning, reasoning, problem-solving, "
        "perception, and decision-making. It is an interdisciplinary field "
        "spanning engineering, mathematics, and computer science. Machine "
        "learning, a core subfield of AI, develops methods and software that "
        "enable machines to perceive their environment and use learning and "
        "intelligence to take actions that maximize their chances of "
        "achieving defined goals. High-profile applications include advanced "
        "web search engines, recommendation systems, generative AI, "
        "autonomous vehicles, and natural language understanding."
    ),
    "Digital twins": (
        "A digital twin is a digital model of an intended or actual "
        "real-world physical product, system, or process that serves as the "
        "effectively indistinguishable digital counterpart of it for "
        "practical purposes such as simulation, integration, testing, "
        "monitoring, and maintenance. A digital twin is a set of adaptive "
        "models that emulate the behaviour of a physical system in a virtual "
        "system, getting real-time data to update itself along its life "
        "cycle. The technology enables monitoring, diagnostics, prognostics, "
        "and optimization across engineering and service domains."
    ),
    "e-Health": (
        "eHealth is the cost-effective and secure use of information and "
        "communications technologies in support of health and health-related "
        "fields, including health-care services, health surveillance, health "
        "literature, and health education. It encompasses electronic health "
        "records, telehealth, mobile health, artificial intelligence in "
        "clinical decision support, and technology-enabled wellness "
        "interventions. The World Health Organization defines eHealth as "
        "including not only internet-based healthcare services but also "
        "modern advancements such as artificial intelligence, wearable "
        "devices, and remote patient monitoring."
    ),
    "Energy": (
        "Energy engineering is a broad field of engineering that deals with "
        "energy efficiency, energy services, facility management, plant "
        "engineering, environmental compliance, sustainable energy, and "
        "renewable energy technologies. It focuses on finding efficient, "
        "clean, and innovative ways to supply, convert, store, and use "
        "energy to meet the world's growing demand in a sustainable manner. "
        "The discipline is concerned with addressing global challenges such "
        "as climate change, carbon reduction, and the transition from fossil "
        "fuels to renewable and sustainable energy sources."
    ),
    "Environment and agriculture": (
        "Environmental science is an interdisciplinary academic field that "
        "integrates physical, biological, and information sciences to study "
        "the environment and find solutions to environmental problems. "
        "Agricultural science is a broad multidisciplinary field of biology "
        "that encompasses the parts of exact, natural, economic, and social "
        "sciences used in the practice and understanding of agriculture. "
        "Together these fields address challenges in food production, "
        "ecosystem management, pollution control, climate-change mitigation, "
        "biodiversity conservation, and the circular bioeconomy."
    ),
    "Information and Communication Technologies": (
        "Information and communications technology (ICT) is an extensional "
        "term for information technology that stresses the role of unified "
        "communications and the integration of telecommunications and "
        "computers, as well as necessary enterprise software, middleware, "
        "storage, and audiovisual systems that enable users to access, "
        "store, transmit, understand, and manipulate information. ICT also "
        "covers the convergence of audiovisual and telephone networks with "
        "computer networks through a single cabling or link system, "
        "including cybersecurity, distributed computing, and cryptographic "
        "protocols."
    ),
    "Medical imaging": (
        "Medical imaging is the technique and process of imaging the "
        "interior of a body for clinical analysis and medical intervention, "
        "as well as visual representation of the function of some organs or "
        "tissues. It seeks to reveal internal structures hidden by the skin "
        "and bones so as to diagnose and treat disease. Medical imaging also "
        "establishes a database of normal anatomy and physiology to make it "
        "possible to identify abnormalities. As a sub-discipline of "
        "biomedical engineering, medical physics, and medicine, it is "
        "concerned with instrumentation, acquisition, reconstruction, "
        "quantitative analysis, and computer-aided detection."
    ),
    "Mobility and Transport": (
        "Transport, or transportation, is the intentional movement of "
        "humans, animals, and goods from one location to another. Modes of "
        "transport include air, land (rail and road), water, cable, "
        "pipeline, and space. The field of transport is important because it "
        "enables communication, trade, and other forms of exchange between "
        "people. Transport research covers vehicle perception and autonomy, "
        "traffic modelling, logistics optimization, electrification, and the "
        "design of multimodal mobility systems."
    ),
    "Natural Language Processing": (
        "Natural language processing (NLP) is the processing of natural "
        "language information by a computer. It is a subfield of computer "
        "science and artificial intelligence that is concerned with giving "
        "computers the ability to understand text and spoken words in much "
        "the same way human beings can. NLP combines computational "
        "linguistics with statistical, machine learning, and deep learning "
        "models. Major tasks include speech recognition, text "
        "classification, natural language understanding, natural language "
        "generation, machine translation, and information extraction."
    ),
    "Quantum and Cryptography": (
        "A quantum computer is a computer that exploits quantum mechanical "
        "phenomena such as superposition and entanglement to perform "
        "computation. Quantum computing intersects deeply with cryptography: "
        "Shor's algorithm showed that a scalable quantum computer could "
        "break widely used public-key cryptosystems, while quantum key "
        "distribution uses entangled quantum states to establish "
        "theoretically unbreakable cryptographic keys. The field spans "
        "quantum hardware, algorithms, error correction, quantum networking, "
        "and the design of post-quantum cryptographic schemes resilient to "
        "quantum attacks."
    ),
    "Therapeutics and Biotechnologies": (
        "Biotechnology is a multidisciplinary field that involves the "
        "integration of natural sciences and engineering sciences in order "
        "to achieve the application of organisms and parts thereof for "
        "products and services. In the therapeutic context, modern "
        "biotechnology has enabled the discovery and manufacturing of "
        "pharmaceutical drugs, gene and cell therapies, mRNA-based "
        "vaccines, diagnostic biomarkers, tissue engineering, and "
        "biomanufacturing processes. Core techniques include genetic "
        "engineering, cell and tissue culture technologies, and emerging "
        "approaches such as CRISPR-based genome editing."
    ),
}


def search_query_for_domain(domain: str, *, use_description: bool = True) -> str:
    """Return the S2 search query string for a given domain."""
    if use_description:
        desc = DOMAIN_DESCRIPTIONS.get(domain, "")
        if desc:
            return f"{domain}: {desc}"
    return domain


def domain_slug(domain: str) -> str:
    """Filesystem-safe slug derived from the domain name."""
    return re.sub(r"[^a-z0-9]+", "_", domain.lower()).strip("_")
