#!/usr/bin/env python
"""
Generate a comprehensive knowledge dataset about Nawab Siraj-ud-Daulah of Bengal.

Creates three files in data/:
  - siraj_narrative.txt   — Rich biographical and historical narrative
  - siraj_qa.txt          — Structured Q&A pairs (User/Metis format)
  - siraj_all.txt         — Combined corpus (narrative + Q&A)

Run:  python data/generate_siraj_dataset.py
"""

import os

# ──────────────────────────────────────────────────────────────────────────────
# NARRATIVE CORPUS — Detailed historical account
# ──────────────────────────────────────────────────────────────────────────────

NARRATIVE = """
Siraj-ud-Daulah was the last independent Nawab of Bengal. He ruled Bengal, Bihar, and Orissa from 1756 to 1757. His full name was Mirza Muhammad Siraj-ud-Daulah. He was born in 1733 in Murshidabad, the capital of Bengal. His mother was Amina Begum and his father was Zainuddin Ahmed Khan. He was the grandson of Nawab Ali Vardi Khan, who raised him as his successor.

Siraj-ud-Daulah became the Nawab of Bengal on April 9, 1756, after the death of his grandfather Ali Vardi Khan. He was only about 23 years old when he ascended the throne. The Bengal Subah was the wealthiest province in the Mughal Empire at that time. It was a center of trade and commerce, especially for textiles, silk, saltpeter, and opium.

The British East India Company had a significant presence in Bengal. They had fortified their settlement in Calcutta without the permission of the Nawab. The company also misused trade privileges called dastaks, which exempted them from paying taxes. This caused great financial loss to the Bengal treasury. Siraj-ud-Daulah saw the British as a threat to his sovereignty.

In June 1756, Siraj-ud-Daulah captured Calcutta from the British East India Company. This was his response to the British fortifying their settlement and abusing trade privileges. The British governor of Calcutta, Roger Drake, and many officials fled the city. The capture of Calcutta led to the infamous Black Hole of Calcutta incident. According to British accounts, 146 British prisoners were confined overnight in a small prison room, and most died from suffocation and heatstroke. However, many historians dispute this account as British propaganda.

Robert Clive arrived from Madras with a British fleet to recapture Calcutta in January 1757. Clive formed a secret alliance with Mir Jafar, who was Siraj-ud-Daulah's commander-in-chief. Mir Jafar was promised the throne of Bengal in exchange for betraying Siraj-ud-Daulah. Other conspirators included Yar Lutuf Khan, Jagat Seth (the richest banker in Bengal), and Omichand.

The Battle of Plassey took place on June 23, 1757, at Palashi near Murshidabad. Siraj-ud-Daulah had about 50,000 soldiers and 53 cannons. The British East India Company had about 3,000 soldiers, of whom only 800 were European. The battle was decided not by military strength but by betrayal. Mir Jafar, who commanded the largest division of the Nawab's army, did not engage in battle. He had secretly agreed to support the British.

Siraj-ud-Daulah lost the Battle of Plassey due to the betrayal of his inner circle. After the battle, he fled Murshidabad on a camel. He was captured by Mir Jafar's soldiers near Rajmahal. On July 2, 1757, Siraj-ud-Daulah was executed on the orders of Mir Jafar's son Miran. He was only about 24 years old at the time of his death.

The Battle of Plassey marked the beginning of British colonial rule in India. Mir Jafar became the new Nawab of Bengal but was a puppet ruler controlled by the British East India Company. The company gained immense wealth from Bengal's treasury and established military supremacy in the region. The battle is considered one of the most important events in Indian history.

Siraj-ud-Daulah is remembered in Bengali history as a young patriot who tried to resist foreign domination. Despite his defeat, he is seen as a symbol of resistance against colonial rule. Many streets and institutions in Bangladesh and West Bengal are named after him. His story is told in Bengali literature, songs, and folklore.

The betrayal at Plassey has become a symbol of treachery in Bengali culture. The phrase "Palashir shorojontro" or the conspiracy of Plassey is well-known. Mir Jafar's name has become synonymous with treachery in Bengali language. The term "mirjafar" is used in Bengali to mean a traitor.

Siraj-ud-Daulah was known for his bravery and youth. He was also criticized for being arrogant and impulsive. He made many enemies among the powerful elites of Bengal, including the Jagat Seth banking family, the Armenian merchants, and his own aunt Ghaseti Begum. These groups conspired with the British to overthrow him.

The economy of Bengal under Siraj-ud-Daulah was very strong. Bengal was known as the richest province of India. It produced fine cotton and silk textiles that were exported worldwide. The British wanted to control Bengal's trade because it was extremely profitable. The revenues from Bengal later helped finance the British Industrial Revolution.

Siraj-ud-Daulah's army included both infantry and cavalry. His forces were equipped with cannons and matchlock muskets. However, his army was not as well-trained as the British forces. The British soldiers had better discipline and modern military tactics. The French East India Company also had a presence in Bengal and sometimes supported the Nawab.

After the Battle of Plassey, the British East India Company became the dominant power in Bengal. They appointed Nawabs who would follow their orders. The company collected taxes, controlled trade, and maintained its own army. This system of indirect rule continued until the company took direct control after the Battle of Buxar in 1764.

The Black Hole of Calcutta remains a controversial historical event. Many modern historians believe the story was exaggerated by the British to justify their conquest of Bengal. Some historians even suggest the incident never happened at all. British historian Thomas Babington Macaulay promoted the story in his writings to portray the British as saviors of India.

Siraj-ud-Daulah's reign was short but historically significant. He ruled for only about 14 months. Yet his defeat changed the course of Indian history permanently. It opened the door for British colonialism that would last nearly 200 years. Understanding Siraj-ud-Daulah is essential to understanding how India came under British rule.

Several important figures were part of the Plassey conspiracy. Jagat Seth was the wealthiest banker in India and his financial support was crucial for the British. Mir Jafar was Siraj-ud-Daulah's military commander who betrayed him for the throne. Robert Clive was the British military commander who led the East India Company forces. William Watts was the British negotiator who arranged the conspiracy with Mir Jafar. Omichand was a wealthy merchant who acted as an intermediary in the conspiracy.

Siraj-ud-Daulah was buried at Khush Bagh in Murshidabad, the cemetery of the Nawabs of Bengal. His tomb is still visited by people who remember his sacrifice. Khush Bagh also contains the graves of other Nawabs of Bengal, including Ali Vardi Khan. The site is maintained by the Archaeological Survey of India.

The Battle of Plassey was fought near the village of Palashi on the banks of the Bhagirathi River. The battlefield was a mango grove surrounded by marshes. This location made it difficult for Siraj-ud-Daulah's large army to maneuver. The British chose the battlefield strategically to neutralize the Nawab's numerical advantage.

Siraj-ud-Daulah's relationship with the Hindu merchant class was strained. He imposed taxes on them and was seen as favoring Muslim nobles. This alienated powerful Hindu bankers and merchants like Jagat Seth and Omichand. They threw their support behind the British because it served their economic interests.

The British victory at Plassey gave them control of Bengal's treasury. They extracted enormous wealth from Bengal through taxes and trade. This wealth was used to fund British military campaigns in other parts of India. It also financed the growing British Empire in Asia. Some historians estimate that Britain extracted billions of dollars worth of wealth from Bengal.

After Plassey, the British East India Company installed Mir Jafar as the puppet Nawab of Bengal. Mir Jafar paid the company large sums of money and granted them extensive trade privileges. When he could not meet their demands, the British replaced him with his son-in-law Mir Qasim. The company continued this pattern of installing and removing puppet rulers.

Siraj-ud-Daulah was among the first Indian rulers to recognize the threat posed by the British East India Company. He tried to limit their power and protect Bengal's sovereignty. His failure was due to internal betrayal rather than military weakness. This pattern of Indian rulers being defeated by betrayal rather than battle would repeat many times in Indian history.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Q&A CORPUS — Structured knowledge in chat format
# ──────────────────────────────────────────────────────────────────────────────

QA_PAIRS = [
    # ── Identity & Background ──
    (
        "Who was Siraj-ud-Daulah?",
        "Siraj-ud-Daulah was the last independent Nawab of Bengal. He ruled Bengal, Bihar, and Orissa from 1756 to 1757. He was the grandson and successor of Nawab Ali Vardi Khan."
    ),
    (
        "What was Siraj-ud-Daulah's full name?",
        "Siraj-ud-Daulah's full name was Mirza Muhammad Siraj-ud-Daulah. He was also sometimes spelled Siraj-ud-Dowla in British records."
    ),
    (
        "When was Siraj-ud-Daulah born?",
        "Siraj-ud-Daulah was born in 1733 in Murshidabad, the capital of Bengal. His mother was Amina Begum and his father was Zainuddin Ahmed Khan."
    ),
    # ── Reign ──
    (
        "When did Siraj-ud-Daulah become Nawab of Bengal?",
        "Siraj-ud-Daulah became the Nawab of Bengal on April 9, 1756, after the death of his grandfather Ali Vardi Khan. He was about 23 years old when he ascended the throne."
    ),
    (
        "How long did Siraj-ud-Daulah rule Bengal?",
        "Siraj-ud-Daulah ruled Bengal for only about 14 months, from April 1756 to July 1757. Despite his short reign, his defeat at the Battle of Plassey changed the course of Indian history."
    ),
    # ── Conflict with British ──
    (
        "Why did Siraj-ud-Daulah conflict with the British East India Company?",
        "Siraj-ud-Daulah conflicted with the British East India Company because they fortified their settlement in Calcutta without his permission and misused trade privileges called dastaks to avoid paying taxes, causing financial loss to Bengal's treasury."
    ),
    (
        "What was the Black Hole of Calcutta?",
        "The Black Hole of Calcutta refers to an incident in June 1756 where British prisoners were allegedly confined in a small prison room overnight after Siraj-ud-Daulah captured Calcutta. Many historians now believe this story was exaggerated by the British as propaganda to justify the conquest of Bengal."
    ),
    (
        "Did Siraj-ud-Daulah capture Calcutta?",
        "Yes, Siraj-ud-Daulah captured Calcutta in June 1756. The British governor Roger Drake and many officials fled the city. This was in response to the British fortifying their settlement without permission."
    ),
    # ── Battle of Plassey ──
    (
        "When was the Battle of Plassey fought?",
        "The Battle of Plassey was fought on June 23, 1757. It took place at Palashi, near Murshidabad, on the banks of the Bhagirathi River in Bengal."
    ),
    (
        "Who fought in the Battle of Plassey?",
        "The Battle of Plassey was fought between Siraj-ud-Daulah, the Nawab of Bengal, and the British East India Company led by Robert Clive. Siraj-ud-Daulah had about 50,000 soldiers while the British had about 3,000 soldiers."
    ),
    (
        "Why did Siraj-ud-Daulah lose the Battle of Plassey?",
        "Siraj-ud-Daulah lost the Battle of Plassey primarily due to betrayal by his commander-in-chief Mir Jafar, who had secretly allied with the British and did not engage his troops in the battle. The battle was decided by treachery rather than military strength."
    ),
    (
        "Who was Robert Clive?",
        "Robert Clive was the British military commander who led the East India Company forces at the Battle of Plassey. He later became the first British Governor of Bengal and is often credited with establishing British rule in India."
    ),
    # ── Betrayal & Death ──
    (
        "Who betrayed Siraj-ud-Daulah?",
        "Siraj-ud-Daulah was betrayed by his own commander-in-chief Mir Jafar, who conspired with the British East India Company. Other conspirators included Yar Lutuf Khan, the banker Jagat Seth, and the merchant Omichand."
    ),
    (
        "How did Siraj-ud-Daulah die?",
        "Siraj-ud-Daulah was executed on July 2, 1757, on the orders of Mir Jafar's son Miran. He was captured near Rajmahal while trying to escape after the Battle of Plassey. He was only about 24 years old."
    ),
    (
        "Where is Siraj-ud-Daulah buried?",
        "Siraj-ud-Daulah is buried at Khush Bagh in Murshidabad, the cemetery of the Nawabs of Bengal. His tomb is still visited today and is maintained by the Archaeological Survey of India."
    ),
    # ── Legacy ──
    (
        "What was the significance of the Battle of Plassey?",
        "The Battle of Plassey marked the beginning of British colonial rule in India. It gave the British East India Company control over Bengal, the wealthiest province of India, and enabled them to establish military supremacy that lasted nearly 200 years."
    ),
    (
        "How is Siraj-ud-Daulah remembered today?",
        "Siraj-ud-Daulah is remembered as a young patriot who tried to resist British colonial rule. Streets and institutions in Bangladesh and West Bengal are named after him. His story is told in Bengali literature, songs, and folklore."
    ),
    (
        "What does the term mirjafar mean in Bengali?",
        "The term mirjafar in Bengali has come to mean a traitor. This is because Mir Jafar, who betrayed Siraj-ud-Daulah at the Battle of Plassey, became synonymous with treachery in Bengali culture."
    ),
    # ── Historical Context ──
    (
        "Who was Mir Jafar?",
        "Mir Jafar was the commander-in-chief of Siraj-ud-Daulah's army who betrayed him at the Battle of Plassey. He became the puppet Nawab of Bengal after the battle, controlled by the British East India Company."
    ),
    (
        "Who was Ali Vardi Khan?",
        "Ali Vardi Khan was the Nawab of Bengal before Siraj-ud-Daulah and his maternal grandfather. He raised Siraj-ud-Daulah as his successor. He died in April 1756 after a reign of 16 years."
    ),
    (
        "What was the Bengal Subah?",
        "The Bengal Subah was the wealthiest province in the Mughal Empire. It included present-day Bangladesh and the Indian state of West Bengal. It was a center of trade in textiles, silk, saltpeter, and opium."
    ),
    (
        "What were dastaks in Bengal?",
        "Dastaks were trade permits that exempted the British East India Company from paying taxes in Bengal. The company misused these permits, causing financial loss to Bengal's treasury and angering Nawab Siraj-ud-Daulah."
    ),
    (
        "Who was Jagat Seth?",
        "Jagat Seth was the title of the wealthiest banker family in Bengal during the time of Siraj-ud-Daulah. The Jagat Seth family financially supported the British East India Company's conspiracy against the Nawab."
    ),
    (
        "What happened to Bengal after the Battle of Plassey?",
        "After the Battle of Plassey, the British East India Company became the dominant power in Bengal. They controlled the treasury, appointed puppet Nawabs like Mir Jafar, collected taxes, and extracted enormous wealth from the region."
    ),
    (
        "How did the British finance their Indian expansion after Plassey?",
        "The British financed their expansion from the wealth they extracted from Bengal after Plassey. They used Bengal's tax revenues and treasury to fund military campaigns across India, eventually establishing control over the entire subcontinent."
    ),
    (
        "What was the conspiracy of Plassey?",
        "The conspiracy of Plassey was a secret alliance between Robert Clive of the British East India Company and Mir Jafar, Siraj-ud-Daulah's military commander. Mir Jafar was promised the throne of Bengal in exchange for betraying the Nawab during battle."
    ),
    (
        "Was Siraj-ud-Daulah a good ruler?",
        "Siraj-ud-Daulah was a young and ambitious ruler who tried to resist British encroachment on Bengal's sovereignty. However, he was criticized for being arrogant and impulsive, which alienated many powerful nobles and merchants who then conspired against him."
    ),
]

# ──────────────────────────────────────────────────────────────────────────────
# GENERATE FILES
# ──────────────────────────────────────────────────────────────────────────────

def generate_qa_text(pairs: list, repetitions: int = 3) -> str:
    """Generate Q&A as User/Metis chat conversations."""
    lines = []
    for _ in range(repetitions):
        for q, a in pairs:
            lines.append(f"User: {q}")
            lines.append(f"Metis: {a}")
            lines.append("")  # blank line between exchanges
    return "\n".join(lines)


def generate_narrative_with_qa(narrative: str, qa_text: str) -> str:
    """Combine narrative and Q&A into a single training corpus."""
    return narrative.strip() + "\n\n\n" + qa_text.strip()


def main():
    os.makedirs("data", exist_ok=True)

    # Clean up the narrative (remove leading/trailing whitespace per paragraph)
    narrative_clean = "\n\n".join(p.strip() for p in NARRATIVE.strip().split("\n\n") if p.strip())

    # Generate Q&A text
    qa_text = generate_qa_text(QA_PAIRS, repetitions=5)

    # Write narrative only
    with open("data/siraj_narrative.txt", "w", encoding="utf-8") as f:
        f.write(narrative_clean)
    print(f"✅ siraj_narrative.txt — {len(narrative_clean):,} characters")

    # Write Q&A only
    with open("data/siraj_qa.txt", "w", encoding="utf-8") as f:
        f.write(qa_text)
    print(f"✅ siraj_qa.txt — {len(qa_text):,} characters")

    # Write combined corpus
    combined = generate_narrative_with_qa(narrative_clean, qa_text)
    with open("data/siraj_all.txt", "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"✅ siraj_all.txt — {len(combined):,} characters (combined corpus)")

    total = len(narrative_clean) + len(qa_text)
    print(f"\n📊 Total dataset: {total:,} characters")
    print(f"   Q&A pairs: {len(QA_PAIRS)} topics × 5 repetitions = {len(QA_PAIRS) * 5} exchanges")
    print("\nReady for training!")
    print("  metis train --dataset data/siraj_all.txt --preset small --iters 10000")


if __name__ == "__main__":
    main()
