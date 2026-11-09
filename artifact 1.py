"""
artifact 1.py - Dataset Generator for Reference-Free RAG Triad Evaluation Task
Generates:
  - input 1.csv : Course Knowledge Base (500 curriculum chunks across Physics, Chemistry, Mathematics, Biology)
  - input 2.csv : Incoming Unlabeled Student Queries (200 realistic test queries without ground truth answers)
"""

import os
import shutil
import random
import pandas as pd
import numpy as np

def generate_datasets():
    output_dir = "/Users/recorder/Downloads/work/task_reference_free_rag_triad"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate / Copy input 1.csv (Knowledge Base)
    src_kb = "/Users/recorder/Downloads/work/task_1w8k865d_rag_doubt_resolution/course_knowledge_base.csv"
    dst_kb = os.path.join(output_dir, "input 1.csv")
    
    if os.path.exists(src_kb):
        shutil.copyfile(src_kb, dst_kb)
        print(f"Copied knowledge base to: {dst_kb}")
    else:
        raise FileNotFoundError(f"Source knowledge base not found at {src_kb}")
    
    kb_df = pd.read_csv(dst_kb)
    print(f"Loaded knowledge base: {len(kb_df)} chunks across {kb_df['subject'].nunique()} subjects.")
    
    # 2. Generate input 2.csv (200 Unlabeled Student Doubts)
    # We create 200 queries spanning:
    #   - 120 Standard In-Domain Doubts (directly coverable by curriculum)
    #   - 50 Boundary / Advanced / Out-of-Scope Doubts (Hallucination bait)
    #   - 30 Conceptual / Ambiguous Doubts
    
    random.seed(42)
    np.random.seed(42)
    
    subjects_chapters = {
        "Physics": ["Thermodynamics", "Optics", "Electromagnetism", "Rotational Dynamics", "Modern Physics"],
        "Chemistry": ["Organic Reaction Mechanisms", "Chemical Equilibrium", "Coordination Chemistry", "Thermodynamics & Energetics", "Electrochemistry"],
        "Mathematics": ["Calculus & Integration", "Linear Algebra & Matrices", "Coordinate Geometry", "Probability & Statistics", "Complex Numbers"],
        "Biology": ["Molecular Biology", "Cell Physiology", "Genetics & Heredity", "Ecology & Environment"]
    }
    
    # Realistic query templates and specific questions
    in_domain_queries = [
        # Physics
        ("Physics", "Thermodynamics", "Why does internal energy remain constant during an isothermal process of an ideal gas?"),
        ("Physics", "Thermodynamics", "What is the physical interpretation of Carnot efficiency and why can it never reach 100%?"),
        ("Physics", "Optics", "How do you calculate the fringe width in Young's double slit experiment when immersed in water?"),
        ("Physics", "Optics", "Why does total internal reflection occur only when light travels from denser to rarer medium?"),
        ("Physics", "Electromagnetism", "How is Lenz's law an expression of the law of conservation of energy?"),
        ("Physics", "Electromagnetism", "What is the magnetic field at the center of a circular coil carrying current I?"),
        ("Physics", "Rotational Dynamics", "Why is the moment of inertia of a hollow cylinder greater than that of a solid cylinder of same mass and radius?"),
        ("Physics", "Rotational Dynamics", "How does angular momentum conservation apply when a spinning figure skater pulls in their arms?"),
        ("Physics", "Modern Physics", "What is the stopping potential in the photoelectric effect and how does it relate to incident frequency?"),
        ("Physics", "Modern Physics", "Explain the de Broglie wavelength formula and how it proves wave-particle duality for electrons."),
        
        # Chemistry
        ("Chemistry", "Organic Reaction Mechanisms", "What is the major product in the elimination reaction of 2-bromobutane with alcoholic KOH?"),
        ("Chemistry", "Organic Reaction Mechanisms", "Why does SN1 substitution lead to partial racemization while SN2 gives complete inversion?"),
        ("Chemistry", "Chemical Equilibrium", "How does increasing pressure affect the equilibrium yield of ammonia in the Haber-Bosch process?"),
        ("Chemistry", "Chemical Equilibrium", "Explain Le Chatelier's principle when an inert gas is added at constant volume."),
        ("Chemistry", "Coordination Chemistry", "Why is [Co(NH3)6]3+ diamagnetic while [CoF6]3- is paramagnetic according to crystal field theory?"),
        ("Chemistry", "Coordination Chemistry", "What is the difference between linkage isomerism and ionization isomerism with examples?"),
        ("Chemistry", "Electrochemistry", "How do you calculate cell potential under non-standard conditions using the Nernst equation?"),
        ("Chemistry", "Electrochemistry", "Why does molar conductivity increase sharply for weak electrolytes upon infinite dilution?"),
        ("Chemistry", "Thermodynamics & Energetics", "What is the condition for a chemical reaction to be spontaneous at all temperatures?"),
        ("Chemistry", "Thermodynamics & Energetics", "Explain the difference between state functions and path functions in chemical thermodynamics."),
        
        # Mathematics
        ("Mathematics", "Calculus & Integration", "How do you evaluate the definite integral of x*sin(x) from 0 to pi using integration by parts?"),
        ("Mathematics", "Calculus & Integration", "What is the geometrical interpretation of Rolle's theorem on a differentiable function?"),
        ("Mathematics", "Linear Algebra & Matrices", "How do you find the eigenvalues and eigenvectors of a 2x2 symmetric matrix?"),
        ("Mathematics", "Linear Algebra & Matrices", "Why is the determinant of an orthogonal matrix always either +1 or -1?"),
        ("Mathematics", "Coordinate Geometry", "What is the condition for two lines y=m1*x+c1 and y=m2*x+c2 to be perpendicular?"),
        ("Mathematics", "Coordinate Geometry", "How do you derive the focal length and eccentricity of an ellipse from its standard equation?"),
        ("Mathematics", "Probability & Statistics", "Explain Bayes' theorem with an example of a medical diagnostic test accuracy."),
        ("Mathematics", "Probability & Statistics", "What is the difference between mutually exclusive events and independent events?"),
        ("Mathematics", "Complex Numbers", "How do you find the cube roots of unity using Euler's formula e^(i*theta)?"),
        ("Mathematics", "Complex Numbers", "What is the locus represented by |z - 3| = |z + 3i| in the complex plane?"),
        
        # Biology
        ("Biology", "Molecular Biology", "What is the role of DNA ligase in the replication of the lagging strand?"),
        ("Biology", "Molecular Biology", "Explain how the lac operon is repressed in E. coli in the absence of lactose."),
        ("Biology", "Genetics & Heredity", "What is the expected phenotypic ratio in a dihybrid cross obeying Mendel's independent assortment?"),
        ("Biology", "Genetics & Heredity", "Why does crossing over during prophase I of meiosis increase genetic variation?"),
        ("Biology", "Cell Physiology", "How does the sodium-potassium ATPase pump maintain resting membrane potential across axons?"),
        ("Biology", "Cell Physiology", "Explain the role of ATP synthase in oxidative phosphorylation during cellular respiration."),
        ("Biology", "Ecology & Environment", "What is the 10 percent rule of energy transfer across trophic levels in an ecosystem?"),
        ("Biology", "Ecology & Environment", "How does primary ecological succession differ from secondary succession after forest fires?")
    ]
    
    # Subtle Out-of-Scope / Advanced / Hallucination Bait Queries (challenging for standard RAG)
    hallucination_bait_queries = [
        # Out-of-scope / fringe physics
        ("Physics", "Thermodynamics", "Can an engine utilizing negative absolute temperature particles exceed the Carnot efficiency limit?"),
        ("Physics", "Optics", "How does negative refractive index metamaterial produce backward Cherenkov radiation?"),
        ("Physics", "Electromagnetism", "Calculate the magnetic monopole flux quantization using Dirac string singularity in curved spacetime."),
        ("Physics", "Rotational Dynamics", "Explain the Dzhanibekov tennis racket instability theorem using non-Euclidean Euler-Poinsot mechanics."),
        ("Physics", "Modern Physics", "How does the holographic principle explain black hole information paradox via Ryu-Takayanagi entanglement entropy?"),
        
        # Fringe chemistry
        ("Chemistry", "Organic Reaction Mechanisms", "Why does benzene undergo spontaneous anti-Markovnikov hydration at room temperature without acid?"),
        ("Chemistry", "Organic Reaction Mechanisms", "Explain the catalytic cycle of Grubbs olefin metathesis involving ruthenium-carbene pi-backbonding."),
        ("Chemistry", "Chemical Equilibrium", "Calculate the fugacity coefficient of supercritical CO2 at 400 bar using Peng-Robinson EOS."),
        ("Chemistry", "Coordination Chemistry", "How does relativistic spin-orbit coupling invert the d-orbital splitting in transuranic actinide complexes?"),
        ("Chemistry", "Electrochemistry", "Explain the Marcus electron transfer reorganization energy in non-adiabatic outer-sphere redox couples."),
        
        # Advanced mathematics
        ("Mathematics", "Calculus & Integration", "How do you rigorously define the functional derivative in infinite-dimensional Banach manifolds?"),
        ("Mathematics", "Calculus & Integration", "State and prove the Selberg trace formula for compact Riemann surfaces of genus g >= 2."),
        ("Mathematics", "Linear Algebra & Matrices", "Explain how spectral decomposition applies to unbounded self-adjoint operators in Hilbert space."),
        ("Mathematics", "Coordinate Geometry", "Derive the Gauss-Bonnet theorem connecting geodesic curvature to Euler characteristic of Riemannian surfaces."),
        ("Mathematics", "Complex Numbers", "Explain how the modular group SL(2, Z) acts on the upper half-plane in elliptic curve j-invariants."),
        
        # Advanced biology / biomedical
        ("Biology", "Molecular Biology", "Describe the molecular architecture of the mammalian spliceosome cryo-EM pre-B complex active site."),
        ("Biology", "Genetics & Heredity", "How does CRISPR-Cas12a collateral cleavage kinetics enable single-molecule DETECTR diagnostic assays?"),
        ("Biology", "Cell Physiology", "Explain the atomic gating mechanism of human Nav1.7 sodium channels via voltage sensor paddle movements."),
        ("Biology", "Ecology & Environment", "Analyze ecological regime shifts using stochastic bifurcation theory and non-local Fisher-KPP equations.")
    ]
    
    # Ambiguous / Low Context Relevance Queries
    ambiguous_queries = [
        ("Physics", "Thermodynamics", "Why is the heat answer coming negative in question 3?"),
        ("Physics", "Optics", "What is the shortcut formula for finding lens magnification?"),
        ("Physics", "Electromagnetism", "Can you explain the right hand thumb rule again in 2 lines?"),
        ("Physics", "Rotational Dynamics", "Why did sir say torque is zero about the hinge?"),
        ("Chemistry", "Organic Reaction Mechanisms", "How to remember all named reactions for organic chemistry exam in 1 day?"),
        ("Chemistry", "Chemical Equilibrium", "What happens if I increase temperature in endothermic reaction?"),
        ("Chemistry", "Coordination Chemistry", "Which color is formed when iron reacts with thiocyanate?"),
        ("Mathematics", "Calculus & Integration", "Please integrate 1/(x^2 + 1) without trigonometric substitution."),
        ("Mathematics", "Linear Algebra & Matrices", "Is rank of product of two matrices always equal to minimum of their ranks?"),
        ("Mathematics", "Probability & Statistics", "Why is coin tossing considered a Bernoulli trial?"),
        ("Biology", "Molecular Biology", "What is the difference between DNA and RNA in short?"),
        ("Biology", "Cell Physiology", "Where is ATP synthesized in cell cytoplasm?")
    ]
    
    records = []
    query_counter = 1
    
    # 1. Expand In-Domain Doubts (~120 queries)
    # Replicate with realistic student variations and phrasing
    variations = [
        "Could you explain: {}",
        "I am confused about: {}",
        "How do we solve doubts regarding: {}",
        "What is the underlying concept behind: {}",
        "{}"
    ]
    
    for i in range(120):
        base_subj, base_chap, base_q = in_domain_queries[i % len(in_domain_queries)]
        var_template = variations[i % len(variations)]
        clean_q = base_q.rstrip("?")
        if "{}" in var_template and var_template != "{}":
            q_text = var_template.format(clean_q.lower()) + "?"
        else:
            q_text = base_q
        
        tier = "Standard" if i % 3 != 0 else "Conceptual"
        records.append({
            "query_id": f"QRY-2026-{query_counter:04d}",
            "student_id": f"STU-{1000 + (query_counter * 17) % 9000:04d}",
            "subject": base_subj,
            "chapter": base_chap,
            "difficulty_tier": tier,
            "query_text": q_text,
            "timestamp": f"2026-10-26 {10 + (query_counter // 25):02d}:{(query_counter * 7) % 60:02d}:{(query_counter * 13) % 60:02d}",
            "session_id": f"SES-{(query_counter * 31) % 500:03d}"
        })
        query_counter += 1
        
    # 2. Expand Hallucination Bait / Out-of-Scope Doubts (~50 queries)
    for i in range(50):
        base_subj, base_chap, base_q = hallucination_bait_queries[i % len(hallucination_bait_queries)]
        records.append({
            "query_id": f"QRY-2026-{query_counter:04d}",
            "student_id": f"STU-{1000 + (query_counter * 17) % 9000:04d}",
            "subject": base_subj,
            "chapter": base_chap,
            "difficulty_tier": "Advanced / Out-of-Scope",
            "query_text": base_q,
            "timestamp": f"2026-10-26 {10 + (query_counter // 25):02d}:{(query_counter * 7) % 60:02d}:{(query_counter * 13) % 60:02d}",
            "session_id": f"SES-{(query_counter * 31) % 500:03d}"
        })
        query_counter += 1
        
    # 3. Expand Ambiguous / Boundary Doubts (~30 queries)
    for i in range(30):
        base_subj, base_chap, base_q = ambiguous_queries[i % len(ambiguous_queries)]
        records.append({
            "query_id": f"QRY-2026-{query_counter:04d}",
            "student_id": f"STU-{1000 + (query_counter * 17) % 9000:04d}",
            "subject": base_subj,
            "chapter": base_chap,
            "difficulty_tier": "Ambiguous / Underspecified",
            "query_text": base_q,
            "timestamp": f"2026-10-26 {10 + (query_counter // 25):02d}:{(query_counter * 7) % 60:02d}:{(query_counter * 13) % 60:02d}",
            "session_id": f"SES-{(query_counter * 31) % 500:03d}"
        })
        query_counter += 1
        
    queries_df = pd.DataFrame(records)
    dst_queries = os.path.join(output_dir, "input 2.csv")
    queries_df.to_csv(dst_queries, index=False)
    print(f"Generated unlabeled student queries: {len(queries_df)} rows saved to: {dst_queries}")
    print(queries_df["difficulty_tier"].value_counts())
    print("Subject distribution:\n", queries_df["subject"].value_counts())

if __name__ == "__main__":
    generate_datasets()
