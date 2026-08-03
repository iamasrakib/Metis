#!/usr/bin/env python
"""
Generate a comprehensive knowledge dataset about cats.

Creates in data/:
  - cat_all.txt — Combined corpus of all cat knowledge (narrative + Q&A)

Run:  python data/generate_cat_dataset.py
"""

import os

# ──────────────────────────────────────────────────────────────────────────────
# CAT FACTS — Narrative knowledge base
# ──────────────────────────────────────────────────────────────────────────────

CAT_FACTS = """
Cats are small carnivorous mammals that have been domesticated for thousands of years. The domestic cat, Felis catus, is a member of the Felidae family. Cats are one of the most popular pets worldwide, valued for their companionship, hunting ability, and unique personalities.

The domestic cat descended from the African wildcat, Felis lybica. This domestication began around 7500 BC in the Near East. Ancient Egyptians were among the first to keep cats as pets, and they revered them as sacred animals. The goddess Bastet was depicted with the head of a cat.

Cats have excellent night vision, which is six times better than humans. Their eyes have a special reflective layer called the tapetum lucidum that enhances vision in low light. Cats also have a wide field of vision of about 200 degrees. However, cats are nearsighted and cannot focus on objects very close to them.

A cat's hearing is extremely sensitive. They can hear sounds up to 64 kHz, which is much higher than the human range of 20 kHz. Cats can rotate their ears 180 degrees independently, allowing them to pinpoint the source of sounds with remarkable accuracy.

Cats have an extraordinary sense of smell, about 14 times more sensitive than humans. They have a specialized organ called the Jacobson's organ, or vomeronasal organ, on the roof of their mouth, which they use to analyze scents. When a cat makes a funny face with its mouth open after smelling something, called a flehmen response, it is using this organ.

A cat's whiskers are highly sensitive tactile hairs called vibrissae. They are embedded deep in the skin and are connected to the nervous system. Whiskers help cats navigate in the dark, judge whether they can fit through openings, and detect changes in air currents. Cats have whiskers on their muzzle, above their eyes, on their chin, and on the back of their front legs.

Cats are obligate carnivores, meaning they must eat meat to survive. Their bodies require nutrients found only in animal tissue. Unlike dogs, cats cannot produce taurine, an essential amino acid, and must get it from their diet. A balanced diet for a cat should be high in protein, moderate in fat, and low in carbohydrates.

Adult cats sleep an average of 12 to 16 hours per day, with some cats sleeping up to 20 hours. Cats are crepuscular, meaning they are most active during dawn and dusk. This is when their natural prey, such as small rodents and birds, are most active.

Cats communicate through a combination of vocalizations, body language, and scent marking. They meow primarily to communicate with humans, not with other cats. Adult cats rarely meow at each other. The meow is a learned behavior that cats develop to get attention from their human companions.

Purring is a low-frequency vibration produced by a cat's laryngeal muscles. Cats purr at a frequency of 25 to 150 Hz. While purring is often associated with contentment, cats also purr when they are stressed, in pain, or giving birth. Some researchers believe purring helps cats heal, as the frequency range promotes bone density and tissue regeneration.

A cat's tongue is covered in tiny backward-facing barbs called papillae. These barbs are made of keratin, the same material as cat claws. The rough tongue helps cats groom themselves efficiently, remove meat from bones, and drink water. Cats lap water by curling their tongue backward and creating a column of liquid that they catch with their mouth.

Cats are known for their agility and balance. They have a flexible spine that allows them to rotate their bodies in mid-air. The righting reflex enables cats to land on their feet when they fall, typically from as little as 30 centimeters. However, high-rise syndrome can occur when cats fall from great heights and sustain injuries.

There are over 70 recognized cat breeds, each with distinct characteristics. The Cat Fanciers' Association (CFA) recognizes 45 breeds. The International Cat Association (TICA) recognizes 73 breeds. Cat breeds can be long-haired or short-haired, and they vary widely in size, temperament, and appearance.

The most popular cat breed worldwide is the Persian, known for its long fur and flat face. The Maine Coon is one of the largest domestic cat breeds, with males weighing up to 18 pounds. The Siamese is known for its distinctive color points and vocal personality. The Bengal cat has a wild appearance with rosette markings reminiscent of a leopard.

Cat colors and patterns are determined by genetics. The most common patterns include solid, tabby, bicolor, tricolor, calico, tortoiseshell, and colorpoint. Tabby cats have distinctive stripes, swirls, or spots, and the tabby pattern is the most common in cats. The gene for orange fur is carried on the X chromosome, which is why most orange cats are male.

Calico cats are almost always female because the coat color gene is on the X chromosome. A calico cat needs two X chromosomes to show both black and orange colors. Male calico cats are extremely rare, occurring in about 1 in 3000 births, and are usually sterile.

Black cats have been both revered and feared throughout history. In ancient Egypt, black cats were considered sacred. In medieval Europe, they were associated with witches and bad luck. In many cultures today, black cats are considered symbols of good luck, particularly in Japan and the United Kingdom.

The average lifespan of an indoor cat is 12 to 18 years, with many cats living into their 20s. The oldest recorded cat, Creme Puff, lived to be 38 years old. Outdoor cats generally have shorter lifespans of 2 to 5 years due to increased risks from traffic, predators, and disease.

Spaying or neutering cats is important for population control and health benefits. It reduces the risk of certain cancers and eliminates unwanted behaviors like spraying and yowling. Cats can be spayed or neutered as early as 8 weeks of age, though many veterinarians recommend 4 to 6 months.

Cats need regular veterinary care, including vaccinations, dental checkups, and parasite prevention. Core vaccines for cats include rabies, feline distemper (panleukopenia), feline herpesvirus, and calicivirus. Annual wellness exams are recommended for adult cats, and semi-annual exams for senior cats over 7 to 10 years old.

Common health problems in cats include dental disease, obesity, diabetes, kidney disease, hyperthyroidism, and urinary tract issues. Dental disease is the most common health problem in cats, affecting up to 85% of cats over three years old. Obesity affects an estimated 60% of domestic cats in developed countries.

Cats are lactose intolerant. Contrary to popular belief, milk can cause digestive upset in most adult cats because they lack the enzyme lactase needed to break down lactose in milk. Fresh water should always be available for cats, and many cats prefer running water from a fountain.

Many common household foods are toxic to cats. These include onions, garlic, chocolate, grapes, raisins, alcohol, caffeine, and xylitol (an artificial sweetener). Certain plants are also toxic, including lilies (which can cause kidney failure), poinsettias, tulips, and sago palm.

Cats are meticulous groomers and spend up to 50% of their waking hours grooming themselves. Regular grooming helps cats regulate body temperature, distribute natural oils, and remove loose fur. Hairballs, or trichobezoars, can occur when cats swallow too much fur during grooming.

Cats are natural hunters and have strong predatory instincts. Even well-fed domestic cats will stalk, pounce, and play with prey. Cats use a technique called the "kill bite" to sever the spinal cord of their prey. Providing interactive toys and play sessions helps satisfy these hunting instincts.

A group of cats is called a clowder. A group of kittens is called a kindle. A male cat is called a tom or tomcat. A female cat is called a queen. A neutered male cat is called a gib. A spayed female cat is simply called a spayed queen. A young cat is called a kitten.

Cats have five toes on their front paws and four toes on their back paws. However, some cats are polydactyl, meaning they have extra toes. The Hemingway cats, descendants of Ernest Hemingway's six-toed cat, are famous polydactyl cats. Polydactyly is more common in certain regions and breeds.

The record for the longest cat ever recorded was a Maine Coon named Stewie, who measured 48.5 inches from nose to tail tip. The heaviest cat on record weighed 46 pounds and 15 ounces. The smallest cat breed is the Singapura, with adult females weighing as little as 4 pounds.

Cats have 30 teeth, while kittens have 26 deciduous (baby) teeth. Cats are born without teeth, grow their baby teeth at about 3 to 4 weeks, and begin losing them at around 3 to 4 months when their permanent teeth come in. A cat's canine teeth are designed for gripping and tearing meat.

A cat's heart beats between 140 and 220 beats per minute, depending on activity level. The normal respiratory rate for a cat is 20 to 30 breaths per minute. A cat's normal body temperature ranges from 100.5 to 102.5 degrees Fahrenheit.

Cats are very flexible and have 30 vertebrae, compared to humans who have 33. Their collarbone is free-floating, which allows them to fit through any space their head can fit through. This is why cats can squeeze through surprisingly small openings.

Cats are fastidious about cleanliness and bury their waste. In the wild, this behavior helps them avoid attracting predators. Modern domestic cats retain this instinct and will generally use a litter box without training. Most cats prefer unscented, clumping litter and a clean box.

The chemical in catnip that attracts cats is called nepetalactone. It affects about 50 to 75 percent of cats and triggers a euphoric response that lasts about 10 to 15 minutes. The sensitivity to catnip is hereditary. Kittens and senior cats are less likely to respond to catnip.

Cats can make over 100 different vocal sounds, while dogs can make only about 10. The most common cat vocalizations include meowing, purring, hissing, growling, chirping, trilling, yowling, and chattering. Cats chirp and chatter when they see birds or other prey through a window.

The Egyptian Mau is believed to be the oldest domestic cat breed, with origins dating back to ancient Egypt. The name Mau actually means cat in ancient Egyptian. The breed is known for its spotted coat and incredible speed, capable of running up to 30 miles per hour.

Japan has a cat-shaped temple called Gotokuji Temple, which is the birthplace of the Maneki-neko or beckoning cat figurine. The Maneki-neko is believed to bring good luck and fortune to its owner. The figurine is often seen in shops and restaurants throughout Japan.

Cats were brought to the Americas by European settlers for pest control on ships. They played a crucial role in protecting food stores from rodents during long voyages. Many of these cats escaped or were released and became the ancestors of feral cat populations in the New World.

Feral cats are domestic cats that have returned to a wild state. They live in colonies and survive without human assistance. Trap-neuter-return (TNR) programs are the most humane and effective method for managing feral cat populations. These programs involve trapping cats, spaying or neutering them, and returning them to their colony.

Cats have a strong territorial instinct. They mark their territory through scent glands on their face, paws, and tail. When a cat rubs its face against furniture or people, it is depositing pheromones to mark them as safe and familiar. Scratching is another way cats mark territory both visually and with scent.

Cats form strong bonds with their human companions. They show affection through slow blinking, head bunting, kneading, and bringing gifts such as toys or prey. A cat's slow blink is often called a "cat kiss" and is a sign of trust and contentment.

Kneading, also called making biscuits, is a behavior that begins in kittenhood. Kittens knead their mother's belly to stimulate milk flow. Adult cats continue this behavior when they feel comfortable and content. It is a sign of happiness and security.

Cats can be trained using positive reinforcement techniques such as clicker training and treats. They respond well to short, frequent training sessions. Cats can learn to perform tricks, walk on a leash, and use a toilet. Training also provides mental stimulation that helps prevent behavior problems.

Interactive play is essential for a cat's physical and mental health. The best toys for cats mimic the movement of prey. Wand toys, laser pointers, puzzle feeders, and treat-dispensing toys are excellent choices. Cats should have at least 15 to 20 minutes of interactive play daily.

Scratching is a natural and necessary behavior for cats. It helps them remove the dead outer layer of their claws, stretch their muscles, and mark territory. Providing appropriate scratching posts made of sisal rope or cardboard can prevent damage to furniture. Scratching posts should be tall enough for cats to fully stretch.

A cat's purr has healing properties. Studies have shown that the frequency range of a cat's purr, 25 to 150 Hz, can improve bone density and promote healing. This frequency range is also used in therapeutic vibration devices for humans. This may explain why cats seem to purr when they are injured or in pain.

Cats can see colors, but not as vividly as humans. They have dichromatic vision, meaning they have two types of cone cells instead of three. Cats see best in the blue-violet and yellow-green spectrums. Reds and pinks appear more gray or green to cats.

Cats are highly sensitive to vibrations and can detect earthquakes before they happen. Some cats can sense changes in barometric pressure and may hide before a storm. Their whiskers can detect minute changes in air currents, helping them navigate in complete darkness.

The ancient Egyptians were the first civilization to domesticate cats. They used cats to protect grain stores from rodents and snakes. Egyptians so revered cats that killing one was a crime punishable by death. Cats were often mummified and buried with their owners to accompany them to the afterlife.

During the Middle Ages in Europe, cats were persecuted due to their association with witchcraft and the devil. This persecution may have indirectly contributed to the spread of the Black Death, as fewer cats meant more rats carrying fleas with the plague bacteria.

In Islam, cats are revered animals. The Prophet Muhammad is said to have had a cat named Muezza. According to Islamic tradition, cats are considered clean animals and are allowed to enter homes and mosques. Feeding and caring for cats is considered a virtuous act in Islamic culture.

Cats are the most popular pet in the United States, with approximately 47 million households owning a cat. There are an estimated 100 million pet cats and 70 million stray cats in the United States. Worldwide, there are over 600 million domestic cats.

A cat's claws are retractable, except for the cheetah. Cats extend their claws for hunting, climbing, and self-defense. When relaxed, their claws are sheathed in the paw to keep them sharp. Cats have five claws on their front paws and four on their back paws, including the dewclaw.

Cats cannot taste sweetness. They lack the receptors for sweet flavors due to a genetic mutation. This is because their natural diet of meat does not contain carbohydrates. Cats can taste sour, bitter, salty, and umami flavors.

The Japanese Bobtail is a breed of cat with a short, rabbit-like tail. These cats are considered good luck in Japan and are often featured in traditional art. The Maneki-neko figurine is typically modeled after a Japanese Bobtail.

Cats spend about 30 to 50 percent of their waking hours grooming themselves. A cat's tongue is covered in tiny hook-like structures called filiform papillae that act like a comb. Grooming helps distribute natural oils throughout the fur, remove loose hair, and regulate body temperature.

The term "cat" comes from the Old English word "catt," which derived from the Latin "catus." In ancient Egyptian, the word was "chaus." The scientific name for the domestic cat, Felis catus, was given by Carl Linnaeus in 1758.

Russian Blue cats are known for their striking silver-blue coat and bright green eyes. They are believed to have originated in the port of Arkhangelsk, Russia. Russian Blues are known for being gentle, intelligent, and somewhat reserved with strangers but very affectionate with their families.

Sphynx cats are one of the few hairless cat breeds. Despite their lack of fur, they are not completely hairless but have a fine down that feels like suede. Sphynx cats require regular bathing to remove oil buildup on their skin. They are known for being extremely affectionate and social.

The Scottish Fold breed is characterized by its distinctive folded ears, which fold forward and downward. This trait is caused by a dominant genetic mutation affecting cartilage. Scottish Folds are known for their sweet temperament and owl-like appearance. Not all cats of this breed have folded ears.

Bengal cats are a hybrid breed developed by crossing domestic cats with the Asian leopard cat. They retain the wild appearance of their ancestors with rosette or marbled patterns. Bengals are highly energetic, intelligent, and require plenty of stimulation. They often enjoy playing in water.

Ragdoll cats are one of the largest domestic cat breeds, with males weighing 12 to 20 pounds. They get their name from their tendency to go limp and relaxed when picked up. Ragdolls have striking blue eyes and a semi-long, silky coat. They are known for their calm and gentle temperament.

The Abyssinian is one of the oldest known cat breeds and resembles the sacred cats of ancient Egypt. They have a distinctive ticked coat where each hair has alternating bands of color. Abyssinians are highly active, curious, and intelligent. They are often described as the clowns of the cat world.

Birman cats are known for their striking blue eyes, white paws, and silky medium-long coat. According to legend, Birmans were the sacred cats of the temples in Burma. They are gentle, quiet, and affectionate without being demanding.

The Norwegian Forest Cat is a large, sturdy breed with a thick, water-resistant coat. They originated in Norway and are adapted to cold climates. In Norse mythology, they were said to be the favorite cats of the goddess Freya. They are friendly, independent, and good climbers.

The Siberian cat is a natural breed from Russia with a semi-long, water-resistant coat. They are known for being hypoallergenic, as they produce lower levels of the Fel d1 protein that causes allergic reactions in humans. Siberians are intelligent, playful, and form strong bonds with their families.

The Cornish Rex has a unique coat consisting only of the downy undercoat, with no guard hairs. This gives them a soft, wavy coat that feels like velvet. Cornish Rex cats are known for their athletic ability, large ears, and affectionate personality. They often seek warmth and enjoy cuddling.

The Manx cat is known for having little to no tail, a result of a natural genetic mutation. They originated on the Isle of Man in the Irish Sea. Manx cats are known for their rounded appearance, powerful hind legs, and gentle temperament. They are excellent jumpers and hunters.

The Chartreux is a rare breed of cat from France, known for its blue-gray coat and copper-colored eyes. According to legend, they were kept by Carthusian monks in French monasteries. Chartreux cats are known for being quiet, intelligent, and excellent hunters.

The Oriental Shorthair is closely related to the Siamese and comes in over 300 color and pattern variations. They have a sleek body, large ears, and an outgoing personality. Orientals are extremely vocal and social, often following their owners around the house.

The Turkish Angora is one of the oldest natural cat breeds, originating in the Ankara region of Turkey. They have a silky, medium-long coat and are often white with blue or amber eyes. Some Turkish Angoras have heterochromia, or two different colored eyes. They are intelligent, playful, and loyal.

Cornish Rex cats have a distinctive egg-shaped head, large ears, and a slender body. Their coat is extremely short and curly. They are known for their high energy levels, intelligence, and people-oriented nature. Cornish Rex cats are sometimes called the greyhounds of the cat world.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Q&A CORPUS — Structured knowledge in chat format
# ──────────────────────────────────────────────────────────────────────────────

QA_PAIRS = [
    # ── Basic Cat Facts ──
    ("What is a domestic cat?", "A domestic cat, Felis catus, is a small carnivorous mammal that has been domesticated for thousands of years. Cats are members of the Felidae family and are one of the most popular pets worldwide. They are valued for their companionship, hunting ability, and unique personalities."),
    ("What did cats descend from?", "Domestic cats descended from the African wildcat, Felis lybica. This domestication began around 7500 BC in the Near East. The African wildcat still exists today and looks very similar to domestic tabby cats."),
    ("How long have cats been domesticated?", "Cats have been domesticated for approximately 9000 years. The earliest evidence of cat domestication dates back to around 7500 BC in the Near East. This is relatively recent compared to dogs, which were domesticated much earlier."),
    ("What is the scientific name of a cat?", "The scientific name of the domestic cat is Felis catus. It was given by Carl Linnaeus in 1758. Cats belong to the kingdom Animalia, phylum Chordata, class Mammalia, order Carnivora, and family Felidae."),

    # ── Cat Anatomy ──
    ("How good is a cat's night vision?", "Cats have excellent night vision that is six times better than humans. They have a special reflective layer called the tapetum lucidum behind their retina that enhances vision in low light. Cats also have a wide field of vision of about 200 degrees. However, cats are nearsighted and cannot focus on objects very close to them."),
    ("How sensitive is a cat's hearing?", "A cat's hearing is extremely sensitive. They can hear sounds up to 64 kHz, which is much higher than the human range of 20 kHz. Cats can rotate their ears 180 degrees independently, allowing them to pinpoint the source of sounds with remarkable accuracy. This helps them detect the high-frequency sounds of small prey like mice."),
    ("How good is a cat's sense of smell?", "Cats have an extraordinary sense of smell, about 14 times more sensitive than humans. They have a specialized organ called the Jacobson's organ on the roof of their mouth, which they use to analyze scents. When a cat makes a funny face with its mouth open after smelling something, called a flehmen response, it is using this organ to get a better sense of the scent."),
    ("What are cat whiskers used for?", "Cat whiskers are highly sensitive tactile hairs called vibrissae. They are embedded deep in the skin and connected to the nervous system. Whiskers help cats navigate in the dark, judge whether they can fit through openings, and detect changes in air currents. Cats have whiskers on their muzzle, above their eyes, on their chin, and on the back of their front legs."),
    ("How many teeth do cats have?", "Adult cats have 30 teeth, while kittens have 26 deciduous or baby teeth. Cats are born without teeth and grow their baby teeth at about 3 to 4 weeks of age. They begin losing their baby teeth at around 3 to 4 months when their permanent teeth come in. A cat's canine teeth are designed for gripping and tearing meat."),
    ("How many bones does a cat have?", "Cats have 230 bones in their body, while humans have 206. A cat has 30 vertebrae, compared to humans who have 33. Their collarbone is free-floating, which allows them to fit through any space their head can fit through, which is why cats can squeeze through surprisingly small openings."),
    ("What is a cat's normal heart rate?", "A cat's heart beats between 140 and 220 beats per minute, depending on activity level. A relaxed cat typically has a lower heart rate, while an active or stressed cat has a higher rate. The normal respiratory rate for a cat is 20 to 30 breaths per minute. A cat's normal body temperature ranges from 100.5 to 102.5 degrees Fahrenheit."),
    ("How many toes does a cat have?", "Cats have five toes on their front paws and four toes on their back paws. However, some cats are polydactyl, meaning they have extra toes. The Hemingway cats, descendants of Ernest Hemingway's six-toed cat, are famous polydactyl cats. Polydactyly is more common in certain regions like the East Coast of the United States."),
    ("Why do cats have rough tongues?", "A cat's tongue is covered in tiny backward-facing barbs called papillae, which are made of keratin, the same material as cat claws. The rough tongue helps cats groom themselves efficiently, remove meat from bones, and drink water. Cats lap water by curling their tongue backward and creating a column of liquid."),
    ("Are cat claws retractable?", "Yes, cat claws are retractable, with the exception of the cheetah. Cats extend their claws for hunting, climbing, and self-defense. When relaxed, their claws are sheathed in the paw to keep them sharp. This retractable mechanism helps cats move silently when stalking prey."),
    ("Can cats see color?", "Cats can see colors, but not as vividly as humans. They have dichromatic vision with two types of cone cells instead of three. Cats see best in the blue-violet and yellow-green spectrums. Reds and pinks appear more gray or green to cats. Their vision is optimized for detecting movement rather than color."),
    ("Do cats have a collarbone?", "Cats have a free-floating collarbone that is not attached to other bones like in humans. This allows them to fit through any space their head can fit through. It also contributes to their incredible flexibility and ability to rotate their bodies in mid-air when falling."),

    # ── Cat Behavior ──
    ("Why do cats purr?", "Purring is a low-frequency vibration produced by a cat's laryngeal muscles at 25 to 150 Hz. While purring is often associated with contentment, cats also purr when stressed, in pain, or giving birth. Research suggests the frequency range promotes bone density and tissue regeneration, so purring may have healing properties."),
    ("Why do cats meow?", "Cats meow primarily to communicate with humans, not with other cats. Adult cats rarely meow at each other. The meow is a learned behavior that cats develop to get attention from their human companions. Different types of meows can indicate different needs such as hunger, greeting, or wanting to go outside."),
    ("Why do cats knead?", "Kneading, also called making biscuits, is a behavior that begins in kittenhood. Kittens knead their mother's belly to stimulate milk flow. Adult cats continue this behavior when they feel comfortable and content. It is a sign of happiness, security, and trust. Cats also have scent glands in their paws, so kneading marks their territory."),
    ("Why do cats sleep so much?", "Adult cats sleep an average of 12 to 16 hours per day, with some cats sleeping up to 20 hours. Cats are crepuscular, meaning they are most active during dawn and dusk. The extensive sleeping is an evolutionary trait that helps them conserve energy for hunting. Despite domestication, cats retain this ancestral behavior."),
    ("What does it mean when a cat slow blinks?", "A cat's slow blink is often called a cat kiss and is a sign of trust and contentment. When a cat slowly blinks at you, it is communicating that it feels safe and comfortable in your presence. You can return the gesture by slowly blinking back at your cat, which helps strengthen your bond."),
    ("Why do cats bring you dead animals?", "Cats bring dead animals to their owners as gifts. This behavior stems from their natural hunting instincts. In the wild, mother cats bring prey to their kittens to teach them how to hunt. Your cat sees you as part of its family and is trying to provide for you or teach you how to hunt."),
    ("Why do cats chatter at birds?", "Cats make a chattering or chirping sound when they see birds or other prey through a window. This behavior may be an expression of frustration at not being able to reach the prey. It might also mimic the killing bite to the neck, as the jaw movements are similar. Some experts believe it is a hunting instinct."),
    ("Why do cats hate water?", "Most domestic cats are not fond of water because their fur becomes heavy when wet, making them uncomfortable and less agile. Their ancestors originated in dry desert regions where they rarely encountered large bodies of water. However, some breeds like the Bengal, Turkish Van, and Maine Coon actually enjoy playing in water."),
    ("Why do cats scratch furniture?", "Scratching is a natural and necessary behavior for cats. It helps them remove the dead outer layer of their claws, stretch their muscles, and mark territory both visually and with scent glands in their paws. Providing appropriate scratching posts made of sisal rope or cardboard can protect furniture."),
    ("What is the flehmen response in cats?", "The flehmen response is when a cat makes a funny face with its mouth open after smelling something. The cat curls its upper lip and opens its mouth to draw air over a specialized organ called the Jacobson's organ on the roof of its mouth. This helps the cat analyze complex scents more thoroughly."),
    ("Why do cats rub against things?", "When a cat rubs its face or body against furniture, objects, or people, it is depositing pheromones from scent glands located on its cheeks, chin, paws, and tail base. This marks the items as safe and familiar territory. When your cat rubs against you, it is claiming you as part of its family."),
    ("Why do cats hide when they are sick?", "Cats instinctively hide when they are sick or in pain because in the wild, showing weakness makes them vulnerable to predators. This survival instinct is so strong that cats often mask signs of illness until they are very sick. This is why regular veterinary checkups are important for early detection of health problems."),

    # ── Cat Health & Nutrition ──
    ("What do cats eat?", "Cats are obligate carnivores, meaning they must eat meat to survive. Their bodies require nutrients found only in animal tissue. A balanced diet for a cat should be high in protein, moderate in fat, and low in carbohydrates. Unlike dogs, cats cannot produce taurine, an essential amino acid, and must get it from their diet."),
    ("Can cats drink milk?", "Despite popular belief, most adult cats are lactose intolerant. They lack the enzyme lactase needed to break down lactose in milk. Feeding milk to cats can cause digestive upset including diarrhea and stomach pain. Fresh water should always be available. Many cats prefer running water from a fountain."),
    ("What foods are toxic to cats?", "Many common foods are toxic to cats. These include onions, garlic, chocolate, grapes, raisins, alcohol, caffeine, and xylitol (an artificial sweetener). Raw eggs, raw fish, and raw meat can also be dangerous due to bacteria. Certain plants including lilies can cause kidney failure and are extremely toxic. Always check with a veterinarian before feeding your cat new foods."),
    ("How often should I feed my cat?", "Adult cats should be fed two meals per day, approximately 12 hours apart. Kittens need more frequent feedings, about three to four meals per day. Portion sizes depend on the cat's age, weight, activity level, and the type of food. Your veterinarian can provide specific feeding recommendations for your individual cat."),
    ("What is the average lifespan of a cat?", "The average lifespan of an indoor cat is 12 to 18 years, with many cats living into their 20s. The oldest recorded cat, Creme Puff, lived to be 38 years old. Outdoor cats generally have shorter lifespans of 2 to 5 years due to increased risks from traffic, predators, and disease."),
    ("How often should a cat see a veterinarian?", "Adult cats should have annual wellness exams. Senior cats over 7 to 10 years old should have semi-annual exams. Kittens need several visits in their first year for vaccinations and health checks. Regular checkups are important even if your cat appears healthy, as cats are masters at hiding illness."),
    ("What vaccines do cats need?", "Core vaccines for cats include rabies, feline distemper (panleukopenia), feline herpesvirus, and calicivirus. Non-core vaccines may include feline leukemia virus and bordetella based on lifestyle and risk factors. Your veterinarian can recommend the appropriate vaccination schedule for your cat."),
    ("Should I spay or neuter my cat?", "Spaying or neutering cats is important for population control and health benefits. It reduces the risk of certain cancers and eliminates unwanted behaviors like spraying and yowling. Cats can be spayed or neutered as early as 8 weeks of age, though many veterinarians recommend 4 to 6 months."),
    ("What are common health problems in cats?", "Common health problems in cats include dental disease, obesity, diabetes, kidney disease, hyperthyroidism, and urinary tract issues. Dental disease is the most common, affecting up to 85% of cats over three years old. Obesity affects about 60% of domestic cats in developed countries. Regular veterinary care can help prevent or manage these conditions."),
    ("How can I tell if my cat is overweight?", "You should be able to feel your cat's ribs with a slight layer of fat over them. Viewed from above, your cat should have a noticeable waist behind the ribs. From the side, the abdomen should tuck up slightly. If your cat has a rounded belly with no waist or you cannot feel the ribs, it may be overweight. Your veterinarian can assess body condition."),
    ("What is dental disease in cats?", "Dental disease is the most common health problem in cats, affecting up to 85% of cats over three years old. It starts with plaque buildup that hardens into tartar, leading to gingivitis, periodontitis, and eventually tooth loss. Signs include bad breath, red gums, drooling, and difficulty eating. Regular dental cleanings and tooth brushing help prevent it."),
    ("What plants are toxic to cats?", "Many common houseplants are toxic to cats. Lilies are extremely dangerous and can cause acute kidney failure even from small exposure. Other toxic plants include poinsettias, tulips, lilies of the valley, sago palm, azaleas, and oleander. If you suspect your cat has eaten a toxic plant, contact your veterinarian immediately."),
    ("Can cats get diabetes?", "Yes, cats can develop diabetes mellitus, especially overweight cats. Feline diabetes is similar to type 2 diabetes in humans. Symptoms include increased thirst, increased urination, weight loss despite increased appetite, and lethargy. Treatment typically involves insulin injections, dietary changes, and weight management."),
    ("What is kidney disease in cats?", "Chronic kidney disease is a common condition in older cats. It occurs when the kidneys gradually lose their ability to filter waste from the blood. Symptoms include increased thirst and urination, weight loss, poor appetite, and lethargy. While not curable, early detection and management can slow progression and maintain quality of life."),
    ("What is hyperthyroidism in cats?", "Hyperthyroidism is a common condition in older cats where the thyroid gland produces too much thyroid hormone. Symptoms include weight loss despite increased appetite, hyperactivity, increased thirst and urination, and a rapid heart rate. Treatment options include medication, dietary therapy, radioactive iodine therapy, or surgery."),
    ("Can cats catch colds from humans?", "Cats generally cannot catch colds from humans because the viruses that cause the common cold in humans are species-specific. However, cats can get upper respiratory infections caused by feline-specific viruses like feline herpesvirus and calicivirus. These are highly contagious between cats but not transmissible to humans."),

    # ── Cat Breeds ──
    ("What is the most popular cat breed?", "The Persian is the most popular cat breed worldwide. Persians are known for their long, luxurious fur and flat face. They have a calm, gentle temperament and are well-suited to indoor living. Persians require regular grooming to prevent matting and keep their coat healthy."),
    ("What is the largest domestic cat breed?", "The Maine Coon is one of the largest domestic cat breeds, with males weighing up to 18 pounds or more. They are known for their large size, tufted ears, bushy tails, and friendly personalities. Despite their size, Maine Coons are gentle and good with children and other pets."),
    ("What are Siamese cats known for?", "Siamese cats are known for their distinctive color points (darker coloring on the ears, face, paws, and tail), striking blue almond-shaped eyes, and vocal personality. They are one of the most talkative cat breeds. Siamese cats are highly social, intelligent, and form strong bonds with their owners."),
    ("What is special about Bengal cats?", "Bengal cats are a hybrid breed developed by crossing domestic cats with the Asian leopard cat. They retain a wild appearance with rosette or marbled coat patterns resembling a leopard. Bengals are highly energetic, intelligent, and require plenty of stimulation. They often enjoy playing in water and are known for their dog-like personalities."),
    ("What are Sphynx cats like?", "Sphynx cats are one of the few hairless cat breeds. Despite their lack of fur, they have a fine down that feels like suede. They require regular bathing to remove oil buildup on their skin. Sphynx cats are known for being extremely affectionate, social, and energetic. They often seek warmth and love to cuddle."),
    ("What is unique about Scottish Fold cats?", "The Scottish Fold breed is characterized by its distinctive folded ears that fold forward and downward. This trait is caused by a dominant genetic mutation affecting cartilage. Scottish Folds are known for their sweet temperament, round faces, and owl-like appearance. Not all cats of this breed have folded ears."),
    ("What are Ragdoll cats known for?", "Ragdoll cats are one of the largest domestic cat breeds, with males weighing 12 to 20 pounds. They get their name from their tendency to go limp and relaxed when picked up. Ragdolls have striking blue eyes and a semi-long, silky coat. They are known for being calm, gentle, and affectionate."),
    ("What is the oldest cat breed?", "The Egyptian Mau is believed to be the oldest domestic cat breed, with origins dating back to ancient Egypt. The name Mau means cat in ancient Egyptian. The breed is known for its naturally spotted coat and incredible speed, capable of running up to 30 miles per hour. Egyptian Maus are loyal, intelligent, and athletic."),
    ("What are Abyssinian cats like?", "Abyssinians are one of the oldest known cat breeds and resemble the sacred cats of ancient Egypt. They have a distinctive ticked coat where each hair has alternating bands of color. Abyssinians are highly active, curious, and intelligent. They are often described as the clowns of the cat world."),
    ("What is the smallest cat breed?", "The Singapura is the smallest cat breed, with adult females weighing as little as 4 pounds. Despite their small size, Singapuras are energetic and playful. They originated in Singapore and have a ticked coat similar to the Abyssinian."),
    ("What is special about Russian Blue cats?", "Russian Blue cats are known for their striking silver-blue coat and bright green eyes. They are believed to have originated in the port of Arkhangelsk, Russia. Russian Blues are known for being gentle, intelligent, and somewhat reserved with strangers but very affectionate with their families."),
    ("What are Birman cats known for?", "Birman cats are known for their striking blue eyes, white paws, and silky medium-long coat. According to legend, they were the sacred cats of temples in Burma. Birmans are gentle, quiet, and affectionate without being demanding. They are sometimes called the sacred cat of Burma."),
    ("What is unique about the Norwegian Forest Cat?", "The Norwegian Forest Cat is a large, sturdy breed with a thick, water-resistant double coat adapted to cold Scandinavian winters. In Norse mythology, they were said to be the favorite cats of the goddess Freya. They are friendly, independent, and excellent climbers."),
    ("Are Siberian cats hypoallergenic?", "Siberian cats are considered hypoallergenic because they produce lower levels of the Fel d1 protein, the main allergen that causes allergic reactions in humans. While no cat is completely hypoallergenic, many allergy sufferers tolerate Siberians well. They originated in Russia and have a semi-long, water-resistant coat."),
    ("What is the Cornish Rex known for?", "The Cornish Rex has a unique coat consisting only of the downy underlayer with no guard hairs, giving them a soft, wavy coat that feels like velvet. They are known for their large ears, slender body, and high energy. Cornish Rex cats are sometimes called the greyhounds of the cat world."),
    ("What is the Manx cat known for?", "The Manx cat is known for having little to no tail, a result of a natural genetic mutation. They originated on the Isle of Man in the Irish Sea. Manx cats have a rounded appearance, powerful hind legs, and are excellent jumpers and hunters. They are known for their gentle temperament."),
    ("What are Oriental Shorthair cats like?", "Oriental Shorthair cats are closely related to the Siamese and come in over 300 color and pattern variations. They have a sleek body, large ears, and an outgoing personality. Orientals are extremely vocal, social, and often follow their owners around the house demanding attention."),
    ("What is the Turkish Angora known for?", "The Turkish Angora is one of the oldest natural cat breeds, originating in the Ankara region of Turkey. They have a silky, medium-long coat and are often white with blue or amber eyes. Some have heterochromia with two different colored eyes. They are intelligent, playful, and loyal."),
    ("What is the Chartreux cat?", "The Chartreux is a rare breed of cat from France, known for its blue-gray coat and copper-colored eyes. According to legend, they were kept by Carthusian monks in French monasteries. Chartreux cats are known for being quiet, intelligent, and excellent hunters."),
    ("What are Japanese Bobtail cats like?", "The Japanese Bobtail is a breed of cat with a short, rabbit-like tail. These cats are considered good luck in Japan and are often featured in traditional art and folklore. The Maneki-neko beckoning cat figurine is typically modeled after a Japanese Bobtail. They are active, intelligent, and sociable."),

    # ── Cat Colors & Patterns ──
    ("What is the most common cat pattern?", "The tabby pattern is the most common cat pattern. Tabby cats have distinctive stripes, swirls, or spots. There are four main tabby patterns: classic (blotched), mackerel (striped), spotted, and ticked. The tabby gene is ancient and can be seen in wild cats as well."),
    ("Why are calico cats almost always female?", "Calico cats are almost always female because the coat color gene is carried on the X chromosome. A calico cat needs two X chromosomes to show both black and orange colors. Male calico cats are extremely rare, occurring in about 1 in 3000 births, and are usually sterile."),
    ("Are black cats bad luck?", "Black cats have been both revered and feared throughout history. In ancient Egypt, black cats were considered sacred. In medieval Europe, they were associated with witches. In many cultures today, black cats are considered good luck, particularly in Japan, the United Kingdom, and parts of Europe."),
    ("Why are most orange cats male?", "The gene for orange fur is carried on the X chromosome. A male cat only needs one copy of the orange gene to be orange, while a female needs two copies. This is why about 80% of orange cats are male. Orange tabbies are sometimes called ginger cats."),
    ("What is a tortoiseshell cat?", "A tortoiseshell or tortie cat has a coat with a mixture of black and orange patches, similar to a tortoise shell. Like calicos, they are almost always female. Tortoiseshell cats without white are often called torties, and those with white patches are called calico or tricolor."),
    ("What is point coloration in cats?", "Point coloration refers to a coat pattern where the body is pale and the extremities (face, ears, paws, and tail) are darker. This pattern is caused by a temperature-sensitive enzyme that produces more pigment in cooler areas of the body. Siamese and Himalayan cats are well-known examples of point coloration."),
    ("What is a tuxedo cat?", "A tuxedo cat is a bicolor cat with a black coat and white markings on the chest, belly, and paws that resemble a tuxedo. They typically have a white blaze or mask on the face. Tuxedo is a color pattern, not a breed, and can occur in many different cat breeds."),

    # ── Cat Communication ──
    ("How do cats communicate with each other?", "Cats communicate through vocalizations, body language, scent marking, and touch. They use different vocalizations including meowing, purring, hissing, growling, and chirping. Body language includes tail position, ear position, posture, and pupil size. Scent marking through rubbing, scratching, and spraying is also key to cat communication."),
    ("What does a cat's tail position mean?", "A cat's tail is a key communication tool. An upright tail indicates confidence and friendliness. A puffed tail indicates fear or aggression. A tail tucked between the legs indicates submission or anxiety. Slow, deliberate tail swishing indicates focused attention. Rapid tail thrashing usually means irritation."),
    ("What does it mean when a cat's ears are flat?", "When a cat flattens its ears against its head, it usually indicates fear, aggression, or defensiveness. This is often called airplane ears. It is a protective posture that keeps the ears safe during a fight. Cats also flatten their ears when they are annoyed or frightened."),
    ("How many vocalizations can cats make?", "Cats can make over 100 different vocal sounds, which is much more than dogs that can only make about 10. Common cat vocalizations include meowing, purring, hissing, growling, chirping, trilling, yowling, and chattering. Each sound has different variations and meanings."),
    ("Why do cats hiss?", "Cats hiss when they feel threatened, scared, or annoyed. Hissing is a defensive warning sound that signals the cat wants to be left alone. It is often accompanied by an arched back, puffed fur, and flattened ears. Hissing is a natural behavior and should not be punished."),
    ("What does cat chattering mean?", "Chattering is a unique vocalization cats make when they see birds, squirrels, or other prey through a window. The sound is a rapid clicking or chirping noise. This may be an expression of excitement and frustration, or it could mimic the killing bite to the neck."),
    ("Do cats talk to each other?", "Cats do communicate with each other, but primarily through body language, scent marking, and touch rather than vocalizations. Adult cats rarely meow at each other. They may hiss, growl, or yowl at each other in tense situations. Friendly cats greet each other with nose touches and mutual grooming."),

    # ── Cat History ──
    ("How were cats domesticated?", "Cats were domesticated through a natural process of self-selection. As humans began farming and storing grain, rodents were attracted to the food stores. Wild cats were drawn to these areas to hunt the rodents. Humans tolerated and eventually welcomed the cats because they provided pest control. Over time, cats became more comfortable living with humans."),
    ("What role did cats play in ancient Egypt?", "Ancient Egyptians revered cats as sacred animals. Cats protected grain stores from rodents and snakes, making them valuable. The goddess Bastet was depicted with the head of a cat. Killing a cat was a crime punishable by death. Cats were often mummified and buried with their owners to accompany them to the afterlife."),
    ("How did cats spread around the world?", "Cats spread around the world primarily through trade and exploration. They were carried on ships for pest control and accompanied traders and explorers. Phoenician ships likely brought cats to Europe. Later, European explorers brought cats to the Americas and Australia. Cats played a crucial role in protecting food stores during long sea voyages."),
    ("What happened to cats in medieval Europe?", "During the Middle Ages in Europe, cats were heavily persecuted due to their association with witchcraft and the devil. They were often killed during religious festivals. This persecution may have indirectly contributed to the spread of the Black Death, as fewer cats meant more rats carrying the plague-infested fleas."),
    ("Are cats mentioned in religion?", "Cats are mentioned in several religions. In Islam, cats are revered animals. The Prophet Muhammad is said to have had a cat named Muezza. Cats are considered clean in Islamic tradition and are allowed to enter mosques. In some Buddhist traditions, cats are also respected. In Norse mythology, the goddess Freya rode a chariot pulled by cats."),
    ("When did cats become popular pets?", "Cats became popular as indoor pets during the Victorian era in the 19th century. The first cat shows were held in London in the 1870s. Since then, selective breeding has produced the wide variety of cat breeds we see today. Cats are now among the most popular pets worldwide."),

    # ── Kitten Care ──
    ("When do kittens open their eyes?", "Kittens are born with their eyes closed and typically begin to open them between 7 to 10 days of age. Their eyes are blue at first and gradually change to their permanent color. All kittens are born with blue eyes, and the adult eye color develops over several weeks."),
    ("How long do kittens need to stay with their mother?", "Kittens should stay with their mother for at least 8 to 12 weeks. During this time, they learn essential social skills, grooming, and how to use the litter box. Early separation can lead to behavioral problems. Kittens are typically weaned by about 8 weeks of age."),
    ("How much do kittens sleep?", "Kittens sleep even more than adult cats, often up to 20 to 22 hours per day. They need this extensive sleep to support their rapid growth and development. When awake, kittens are highly active and playful, exploring their environment and learning important skills."),
    ("What do kittens eat?", "Kittens should eat a specially formulated kitten food that is higher in protein, fat, and calories than adult cat food. They need these extra nutrients to support their rapid growth. Kittens should be fed three to four meals per day until they are about six months old, then transition to two meals per day."),
    ("When should a kitten first see a veterinarian?", "A kitten should have its first veterinary visit within a few days of being adopted, typically at around 8 weeks of age. The first visit usually includes a health examination, initial vaccinations, deworming, and testing for feline leukemia and FIV. The veterinarian will also discuss spaying or neutering."),
    ("How to litter train a kitten?", "Litter training a kitten is usually instinctive. Place the kitten in a clean litter box after meals, naps, and play sessions. Use a shallow box with unscented litter. Most kittens will naturally use the box. Keep the litter box clean by scooping daily. Never punish accidents, as this can create negative associations."),

    # ── Senior Cat Care ──
    ("When is a cat considered senior?", "Cats are generally considered senior at 7 to 10 years of age. Cats over 10 years are often called geriatric. With advances in veterinary care, more cats are living well into their teens and even twenties. Senior cats need more frequent veterinary checkups, typically every six months."),
    ("What health issues are common in senior cats?", "Common health issues in senior cats include kidney disease, hyperthyroidism, diabetes, arthritis, dental disease, vision and hearing loss, cognitive decline, and cancer. Regular veterinary checkups are essential for early detection and management of these conditions."),
    ("How should I care for my senior cat?", "Senior cat care includes more frequent veterinary visits (every 6 months), a diet formulated for senior cats, joint supplements like glucosamine, easy access to food and water, comfortable bedding, keeping them indoors, and monitoring for changes in behavior, appetite, or litter box habits."),
    ("Do senior cats need special food?", "Many senior cats benefit from a diet formulated for older cats. Senior cat foods often have adjusted protein levels, added joint supplements like glucosamine and chondroitin, omega-3 fatty acids for joint and kidney health, and antioxidants to support the immune system."),
    ("What is feline cognitive dysfunction?", "Feline cognitive dysfunction is similar to Alzheimer's disease in humans. It affects older cats and causes confusion, disorientation, changes in sleep patterns, decreased social interaction, and house-soiling. While not curable, management strategies include maintaining routines, environmental enrichment, and sometimes medication."),

    # ── Famous Cats ──
    ("Who was the oldest cat ever recorded?", "The oldest recorded cat was Creme Puff, who lived to be 38 years old and 3 days. She lived in Austin, Texas, with her owner. The second oldest cat on record was a tabby named Grandpa who lived to be 34. Most indoor cats live to 12 to 18 years."),
    ("What is the longest cat on record?", "The longest cat ever recorded was a Maine Coon named Stewie, who measured 48.5 inches from nose to tail tip. He lived in Nevada and passed away in 2013. The record for the longest tail on a domestic cat is 19 inches, held by a Maine Coon named Cygnus."),
    ("Who was Grumpy Cat?", "Grumpy Cat, whose real name was Tardar Sauce, was an internet celebrity cat known for her permanently grumpy facial expression caused by feline dwarfism and an underbite. She became one of the most famous internet cats, generating millions of dollars in merchandise. She lived from 2012 to 2019."),
    ("Who was Maru the cat?", "Maru was a Scottish Fold cat from Japan who became famous on YouTube for his love of playing in cardboard boxes. His channel had hundreds of millions of views and he became one of the most famous cats on the internet. Maru helped popularize Scottish Fold cats worldwide."),
    ("What is the Maneki-neko?", "The Maneki-neko, or beckoning cat, is a common Japanese figurine believed to bring good luck and fortune. It typically depicts a cat with one paw raised in a beckoning gesture. The figurine is often seen in shops and restaurants throughout Japan and is sometimes called the lucky cat or fortune cat."),
    ("What cats did Ernest Hemingway keep?", "Ernest Hemingway was given a six-toed (polydactyl) cat named Snow White by a ship captain. He kept many polydactyl cats at his home in Key West, Florida. Today, the Hemingway Home and Museum is home to about 50 descendants of his original cats, about half of which are polydactyl. These cats are often called Hemingway cats."),

    # ── Cat Care ──
    ("How often should I groom my cat?", "Short-haired cats should be brushed weekly to remove loose fur and distribute natural oils. Long-haired cats need daily brushing to prevent mats and tangles. All cats benefit from regular grooming, which reduces hairballs, stimulates circulation, and is a bonding opportunity with your cat."),
    ("How often should I clean the litter box?", "The litter box should be scooped at least once daily, and ideally twice daily. The entire litter box should be emptied, washed with mild soap, and refilled with fresh litter every 1 to 2 weeks. Cats prefer unscented, clumping litter. A clean litter box encourages proper litter box use."),
    ("How many litter boxes should I have?", "The general rule is one litter box per cat plus one extra. For example, a household with two cats should have three litter boxes. Litter boxes should be placed in quiet, accessible locations away from food and water. Multiple boxes help prevent territorial issues and give cats options."),
    ("How do I trim my cat's nails?", "Use cat-specific nail clippers and trim only the white tip of the nail, avoiding the pink quick that contains blood vessels and nerves. If your cat resists, try trimming one paw at a time with treats and praise. Scratching posts also help maintain nail health."),
    ("Should I bathe my cat?", "Most cats do not need regular baths because they are meticulous self-groomers. However, some situations require baths, such as if the cat gets into something sticky or toxic, or for hairless breeds like the Sphynx that need regular bathing to remove skin oil. When bathing, use cat-specific shampoo and avoid getting water in the ears."),
    ("How do I introduce a new cat to my home?", "Introduce a new cat gradually. Start by keeping the new cat in a separate room with its own food, water, litter box, and bed. Allow the cats to smell each other under the door. After a few days, do short supervised meetings. Use positive reinforcement like treats. Full integration can take weeks to months."),
    ("What toys do cats like best?", "Cats prefer toys that mimic the movement of prey. Wand toys with feathers, laser pointers, toy mice, and crinkle balls are popular. Puzzle feeders and treat-dispensing toys provide mental stimulation. Rotate toys regularly to maintain interest. Interactive play sessions are more important than leaving toys out."),
    ("How much playtime does a cat need?", "Cats should have at least 15 to 20 minutes of interactive playtime daily, ideally split into two sessions. Play sessions should mimic hunting behaviors with stalking, chasing, pouncing, and catching. Interactive play helps prevent obesity, reduces behavioral problems, and strengthens the bond between you and your cat."),

    # ── Cat Training ──
    ("Can cats be trained?", "Yes, cats can be trained using positive reinforcement techniques like clicker training and treats. They respond best to short, frequent training sessions of 5 to 10 minutes. Cats can learn tricks like sit, high five, and fetch, as well as more practical skills like walking on a leash and using a cat flap."),
    ("How do I stop my cat from scratching furniture?", "Provide appealing alternatives like tall scratching posts covered in sisal rope or corrugated cardboard. Place scratching posts near the furniture your cat is scratching. Use positive reinforcement when your cat uses the post. Never punish your cat for scratching. You can also use soft nail caps and furniture protectors."),
    ("How do I stop my cat from biting?", "Kittens learn bite inhibition from their littermates and mother. If an adult cat bites, it may be due to overstimulation, fear, or play aggression. Watch for warning signs like tail thrashing or ear flattening. Stop interacting when the cat shows signs of overstimulation. Never use physical punishment, which can increase aggression."),
    ("Can cats walk on a leash?", "Yes, many cats can be trained to walk on a leash and harness. Use a cat-specific harness that fits snugly, never a collar alone. Start by letting the cat wear the harness indoors for short periods. Then attach the leash and let the cat drag it around. Finally, practice walking together in a quiet area. Always be patient."),
    ("How do I stop my cat from meowing excessively?", "First, rule out medical issues with a veterinarian. Excessive meowing can indicate pain, anxiety, or underlying health problems. Ensure the cat's basic needs are met. Do not reward attention-seeking meows by responding. Only give attention when the cat is quiet. Provide enrichment and playtime to reduce boredom."),

    # ── Cat Products ──
    ("What type of litter box is best?", "The best litter box depends on your cat. Most cats prefer open, uncovered boxes that are large enough to turn around in. Hooded boxes trap odors but some cats feel trapped. Self-cleaning boxes are convenient but can scare some cats. The most important factor is keeping the box clean."),
    ("What is the best cat food?", "The best cat food is nutritionally complete and appropriate for your cat's life stage. Look for foods with high-quality animal protein as the first ingredient. Wet food provides hydration and is closer to a cat's natural diet. Dry food is convenient. A combination of both can offer benefits. Always consult your veterinarian."),
    ("What cat bed should I get?", "Cats like beds that are cozy, warm, and in a safe location. Enclosed beds or cave-style beds provide security. Heated beds are popular, especially for senior cats. Some cats prefer simple cardboard boxes. The best bed is one your cat actually uses, so observe where your cat already likes to sleep."),
    ("Are cat water fountains good?", "Yes, cat water fountains are excellent because many cats prefer running water. Fountains encourage increased water intake, which benefits kidney and urinary tract health. The constant circulation also keeps water fresh and oxygenated. Stainless steel or ceramic fountains are easier to keep clean than plastic ones."),

    # ── Cat Myths and Misconceptions ──
    ("Do cats always land on their feet?", "Cats have a remarkable righting reflex that allows them to orient themselves and land on their feet when falling. However, they do not always land safely. High-rise syndrome can cause serious injuries when cats fall from significant heights. Cats can also injure themselves from relatively low falls."),
    ("Are cats aloof and unfriendly?", "This is a common myth. Cats form strong bonds with their human companions. They show affection through slow blinking, head bunting, kneading, rubbing, and spending time near their people. Each cat has a unique personality, and some are more affectionate than others."),
    ("Can cats see in complete darkness?", "Cats cannot see in complete darkness. They need some light to see. However, their vision in low light is excellent, about six times better than human vision. They have a reflective layer called the tapetum lucidum that enhances their ability to see in dim light."),
    ("Do cats hate other cats?", "Cats are naturally solitary hunters but can form social bonds with other cats. Some cats enjoy living with feline companions, especially if raised together or properly introduced. Others prefer to be the only cat in the household. A cat's social preference depends on their personality and early socialization."),
    ("Is it true that cats steal babies' breath?", "No, this is a harmful myth. Cats do not steal babies' breath. This superstition likely originated from confusion about Sudden Infant Death Syndrome. However, it is still recommended to supervise interactions between cats and infants for safety reasons, and keep cats out of the crib."),

    # ── Cat Body Language ──
    ("What does it mean when a cat's tail is puffed up?", "A puffed up or bristled tail indicates fear, agitation, or defensiveness. The fur stands on end to make the cat appear larger and more intimidating to potential threats. This is often accompanied by an arched back and is a sign that the cat is frightened and may become defensive."),
    ("What does it mean when a cat rolls over and shows its belly?", "When a cat rolls over and shows its belly, it is a sign of trust and relaxation. However, unlike dogs, this is not always an invitation for belly rubs. Many cats will attack if you touch their belly. The exposed belly means your cat feels safe in its environment."),
    ("What does it mean when a cat's pupils are dilated?", "Dilated pupils in cats can indicate several things including excitement, fear, playfulness, or low light. Context is important. Dilated pupils during play mean the cat is excited. In a stressful situation, dilated pupils indicate fear. Always consider the cat's overall body language to understand the full picture."),
    ("What does it mean when a cat arches its back?", "An arched back can mean different things. A Halloween cat pose with puffed fur and arched back indicates fear and defensiveness. A cat stretching with an arched back and a yawn indicates relaxation. During play, an arched back with sideways hopping is an invitation to play."),
    ("How do cats show affection?", "Cats show affection in many ways: slow blinking (cat kisses), head bunting (rubbing their head against you), kneading (making biscuits), purring, following you around, bringing gifts, sitting on you or near you, grooming you, and exposing their belly. Each cat has a unique way of expressing love."),

    # ── Additional Cat Facts ──
    ("What is a group of cats called?", "A group of cats is called a clowder. A group of kittens is called a kindle. A male cat is called a tom or tomcat. A female cat is called a queen. A neutered male cat is called a gib. A young cat is called a kitten."),
    ("How fast can a cat run?", "The average domestic cat can run at speeds of about 30 miles per hour over short distances. This speed helps them catch prey and escape predators. Cat's powerful hind legs and flexible spine contribute to their running ability."),
    ("How high can a cat jump?", "Cats can jump up to six times their body length in a single leap. This remarkable jumping ability comes from their powerful hind leg muscles and flexible spine. A typical cat can jump about 5 to 6 feet high from a standing position."),
    ("What is catnip and how does it work?", "Catnip is a plant called Nepeta cataria that contains an oil called nepetalactone. When cats inhale nepetalactone, it triggers a euphoric response that lasts about 10 to 15 minutes. About 50 to 75 percent of cats are affected by catnip. The sensitivity is hereditary, and kittens and senior cats are less likely to respond."),
    ("How much water should a cat drink?", "A cat should drink approximately 4 ounces of water per 5 pounds of body weight per day. For a 10-pound cat, that is about 8 ounces or one cup of water daily. Cats on wet food get more water from their diet than cats on dry food."),
    ("Can cats get separation anxiety?", "Yes, cats can develop separation anxiety. Signs include excessive meowing, destructive behavior, inappropriate elimination, and clinginess when you are home. Providing enrichment, interactive toys, and a consistent routine can help. In severe cases, consult a veterinarian or animal behaviorist."),
    ("Do cats dream?", "Yes, cats dream just like humans. Studies of feline brain activity show that cats experience REM sleep, the stage where dreaming occurs. Kittens and senior cats tend to have more REM sleep. You may notice your cat twitching, moving its paws, or making sounds while sleeping."),
    ("What is the best way to pick up a cat?", "The safest way to pick up a cat is to place one hand under the chest behind the front legs and the other hand under the hindquarters for support. Lift gently and hold the cat close to your body. Never lift a cat by the scruff of the neck, as this can cause pain and injury."),
    ("Why do cats like boxes?", "Cats love boxes because they provide security, warmth, and a hiding spot. A box gives cats a safe place to observe their environment without being seen. Boxes also provide insulation and stress relief. This behavior is instinctual for cats who naturally seek enclosed spaces."),
    ("Why do cats sit in small spaces?", "Cats feel secure in small, enclosed spaces that allow them to hide from potential threats. Small spaces also trap body heat, keeping cats warm. This behavior comes from their wild ancestors who sought out dens and crevices for protection from predators and the elements."),

    # ── Cat Behavior Problem Solving ──
    ("Why is my cat spraying urine?", "Spraying or urine marking is a territorial behavior where cats back up to vertical surfaces and spray small amounts of urine. Causes include territorial disputes, stress, unneutered status, and changes in the household. Spaying or neutering reduces spraying in 90% of cats. Reducing stress, increasing resources, and cleaning marked areas with enzymatic cleaners can help."),
    ("Why is my cat aggressive?", "Cat aggression can stem from fear, territoriality, play aggression, redirected aggression, or pain. Always rule out medical causes first. Treatment depends on the type of aggression and includes behavior modification, environmental changes, and sometimes medication. Consult a veterinarian or board-certified veterinary behaviorist for serious cases."),
    ("Why is my cat not using the litter box?", "Inappropriate elimination has many causes: medical issues like UTIs, litter box aversion, location preference, substrate preference, or stress. First, have your cat examined by a veterinarian to rule out medical problems. Ensure the litter box is clean, accessible, and in a quiet location. Use unscented litter and have enough boxes (one per cat plus one)."),
    ("Why does my cat wake me up at night?", "Cats are crepuscular, meaning they are most active at dawn and dusk. To stop nighttime waking, provide plenty of interactive play before bed, feed a meal at night, ignore the cat when it wakes you (don't reward the behavior), and provide enrichment toys. Automatic feeders can help with early morning feeding."),
    ("Why does my cat eat plastic or other non-food items?", "Eating non-food items, called pica, can indicate nutritional deficiencies, medical issues like anemia, or behavioral problems like stress or boredom. Some cats are attracted to certain textures. Provide appropriate chew toys, cat grass, and mental stimulation. If the behavior persists, consult a veterinarian."),
    ("How do I deal with a hyperactive cat?", "Hyperactive cats need more physical and mental stimulation. Provide multiple interactive play sessions daily, puzzle feeders, cat trees for climbing, window perches, and rotating toys. Consider clicker training for mental exercise. Some cats, especially Bengals and Siamese, are naturally high-energy and need extra enrichment."),
    ("Why does my cat follow me everywhere?", "Following you everywhere is usually a sign of affection and bonding. Your cat sees you as a source of safety, comfort, and positive experiences. Some breeds like Siamese and Ragdolls are more likely to shadow their owners. It can also indicate curiosity, hunger, or a desire for attention."),
    ("How can I reduce my cat's stress?", "Reduce cat stress by providing a predictable routine, plenty of vertical space, hiding spots, and multiple resources (food, water, litter boxes) in multi-cat households. Use synthetic pheromone diffusers like Feliway. Provide enrichment through play, puzzle toys, and window access. Gentle, predictable interactions also help."),
    ("What is single kitten syndrome?", "Single kitten syndrome refers to behavioral issues that can develop in kittens raised without littermates. These kittens may not learn proper bite inhibition, play too rough, or develop separation anxiety. Adopting kittens in pairs is often recommended to help them learn social skills from each other."),

    # ── Cat Safety & First Aid ──
    ("What should I do in a cat emergency?", "In a cat emergency, stay calm and contact your veterinarian or the nearest emergency animal hospital immediately. Common emergencies include difficulty breathing, severe bleeding, poisoning, seizures, heatstroke, and trauma. Learn basic cat first aid and keep an emergency kit with your vet's number, a pet first aid book, and supplies."),
    ("What human medications are dangerous for cats?", "Many human medications are dangerous for cats, including acetaminophen (Tylenol), ibuprofen (Advil), aspirin, antidepressants, and sleep aids. Acetaminophen is especially toxic and can be fatal. Never give your cat any human medication without veterinary approval. Keep all medications securely stored out of reach."),
    ("How do I cat-proof my home?", "Cat-proof your home by securing loose electrical cords, removing toxic plants, keeping small objects that could be swallowed out of reach, ensuring windows have secure screens, keeping cleaning products and medications in cabinets, covering sharp edges, and checking appliances (dryers, washers) before use."),
    ("What should I include in a cat first aid kit?", "A cat first aid kit should include: your vet's phone number and the nearest emergency vet, gauze pads and bandages, blunt-tip scissors, tweezers, an oral syringe for giving medications, styptic powder for bleeding nails, antiseptic solution (chlorhexidine), a digital thermometer, a towel or blanket for restraint, and a pet carrier."),
    ("What temperature is dangerous for cats?", "Cats can overheat in temperatures above 100°F (38°C) and can get hypothermia below 45°F (7°C). Never leave cats in parked cars, as temperatures can become deadly within minutes. Provide cool, shaded areas and plenty of water in summer. In winter, provide warm bedding and keep cats indoors."),
    ("How do I know if my cat has a fever?", "Normal cat temperature is 100.5 to 102.5°F (38 to 39.2°C). Signs of fever include warm ears, lethargy, loss of appetite, shivering, and hiding. The only accurate way to measure is with a rectal digital thermometer. If you suspect a fever, consult your veterinarian."),
    ("What are signs of poisoning in cats?", "Signs of poisoning include drooling, vomiting, diarrhea, difficulty breathing, seizures, lethargy, weakness, pale gums, and dilated pupils. If you suspect poisoning, contact your veterinarian or a pet poison helpline immediately. Do not induce vomiting unless directed by a professional."),

    # ── Traveling with Cats ──
    ("How do I travel with my cat in a car?", "Always use a secure, well-ventilated cat carrier in the car. Never let a cat roam freely in a moving vehicle. Place the carrier on the back seat secured with a seatbelt. Cover the carrier with a light blanket to reduce stress. Take breaks on long trips to offer water. Never leave a cat alone in a parked car."),
    ("How do I help my cat adjust to a new home?", "Help a cat adjust by setting up a small safe room with food, water, litter box, bed, and toys. Let the cat explore this room first. Gradually introduce the rest of the house over several days. Maintain a consistent routine. Use pheromone diffusers, and give the cat plenty of patience and positive reinforcement."),
    ("Can I travel on a plane with my cat?", "Yes, cats can travel on planes either in-cabin (if small enough) or as checked cargo. In-cabin travel requires an airline-approved carrier that fits under the seat. You'll need a health certificate from your veterinarian issued within 10 days of travel. Book early as airlines limit pet spots. Never sedate your cat for air travel without veterinary approval."),
    ("Should I board my cat or use a pet sitter?", "Most cats prefer to stay in their own home with a pet sitter rather than being boarded. Cats are territorial and become stressed in unfamiliar environments. A reliable pet sitter who visits once or twice daily can provide food, water, litter box cleaning, play, and medication if needed. Boarding is acceptable if your cat is comfortable with it."),

    # ── Multi-Cat Households ──
    ("How do I introduce cats to each other?", "Introduce cats gradually over 1-3 weeks. Keep the new cat in a separate room initially. Swap scents by exchanging bedding. Allow visual contact through a crack in the door. Then do short supervised meetings with treats and positive reinforcement. Never force interactions. Some cats may take months to fully accept each other."),
    ("How many cats is too many?", "The appropriate number of cats depends on your space, resources, and ability to provide care. Most experts recommend no more than 4-6 cats in a typical home. More cats require more resources: litter boxes (n+1 rule), food stations, water sources, vertical space, and attention. Each cat needs their own space and resources."),
    ("Why do my cats fight?", "Cats may fight due to territorial disputes, redirected aggression, play aggression that escalates, or personality conflicts. Ensure adequate resources are spread throughout the home. Use positive reinforcement for calm behavior. In severe cases, consult a veterinary behaviorist. Never physically intervene in a cat fight."),
    ("Can cats be friends with other pets?", "Many cats can live peacefully with dogs, especially if introduced properly and raised together. Early socialization is key. Some cats also bond with rabbits, guinea pigs, and other small pets, though supervision is essential as cats are natural predators. Always supervise initial interactions and provide escape routes for smaller animals."),

    # ── Cat Adoption ──
    ("What should I consider before adopting a cat?", "Consider your lifestyle, living situation, budget, and commitment. Cats can live 15-20 years. Costs include food, litter, veterinary care, and supplies. Consider whether your home allows pets, if anyone has allergies, and if you have time for daily play and interaction. Adoption from shelters saves lives and often includes initial veterinary care."),
    ("Should I adopt a kitten or an adult cat?", "Kittens are adorable but require more time, training, and supervision. Adult cats have established personalities and are often calmer. Shelters commonly have adult cats waiting for homes. Adult cats may already be litter trained and spayed or neutered. Consider your energy level and available time when deciding."),
    ("What questions should I ask at a cat shelter?", "Ask about the cat's history, temperament, health status, vaccinations, whether it's spayed/neutered, its behavior with other animals and children, and why it was surrendered if applicable. Also ask about any known medical issues or behavioral concerns. A good shelter will be transparent about the cat's needs."),
    ("How much does it cost to own a cat per year?", "Annual cat ownership costs typically range from $500 to $2,000. This includes food ($200-500), litter ($100-300), routine veterinary care including vaccinations and checkups ($150-300), pet insurance or emergency fund ($200-600), toys, treats, and supplies ($100-300). Initial setup costs for a new cat are higher."),
    ("Is pet insurance worth it for cats?", "Pet insurance can be valuable for covering unexpected veterinary costs. Typical plans cover accidents and illnesses, while wellness plans cover routine care. Premiums range from $15-40 per month. Consider your cat's risk factors, breed-specific health issues, and your financial situation. Insurance is best purchased when the cat is young and healthy."),

    # ── Cat-Friendly Environment ──
    ("What are cat-safe plants?", "Cat-safe plants include catnip, cat grass (wheatgrass), spider plants, Boston ferns, African violets, areca palms, and bamboo. Always check with the ASPCA toxic plant list before bringing plants into a cat household. Provide cat grass for your cat to nibble on as a safe alternative to houseplants."),
    ("How do I create a cat-friendly home?", "Create a cat-friendly home with vertical space (cat trees, shelves, window perches), hiding spots (boxes, cat caves), scratching posts, multiple water stations, window access for bird watching, interactive toys, puzzle feeders, and comfortable sleeping areas. Cats need territory they can navigate and control."),
    ("What is catification?", "Catification is a term coined by cat behaviorist Jackson Galaxy for designing your home to meet your cat's natural needs. It involves creating vertical territory, cat superhighways along walls or shelves, cozy hiding spots, window perches, and dedicated play areas. Catification prevents behavioral problems by satisfying cats' instincts."),
    ("Why do cats need vertical space?", "Vertical space is essential for cats because they are vertical territorial animals. Cat trees, shelves, and perches allow cats to observe their territory from above, escape from other pets or children, and feel secure. Vertical space is especially important in multi-cat households to reduce competition for territory."),
    ("Should cats be kept indoors?", "Indoor cats live significantly longer (12-18 years vs 2-5 years for outdoor cats) and are protected from traffic, predators, diseases, and parasites. However, indoor cats need enrichment to prevent boredom. If you want your cat to experience outdoors, consider leash training or building a catio (enclosed outdoor space)."),

    # ── Seasonal Cat Care ──
    ("How do I care for my cat in summer?", "In summer, provide plenty of fresh water, keep your cat indoors during the hottest parts of the day, use fans or air conditioning, provide cool surfaces, brush more frequently to remove loose fur, watch for signs of heatstroke like panting and lethargy, and never leave your cat in a parked car."),
    ("How do I care for my cat in winter?", "In winter, provide warm bedding away from drafts, maintain a consistent indoor temperature, check under the hood of your car before starting (cats may hide there for warmth), use pet-safe ice melts, wipe your cat's paws after walks, and provide extra enrichment for days when your cat is stuck indoors."),
    ("Can cats get sunburned?", "Yes, cats can get sunburned, especially white cats, hairless cats, and cats with light-colored ears and noses. Sunburn can lead to skin cancer, particularly squamous cell carcinoma. Protect cats by limiting sun exposure during peak hours, using pet-safe sunscreen, and providing shaded areas."),

    # ── Cat Genetics ──
    ("What determines a cat's fur color?", "Cat fur color is determined by genetics involving multiple genes. The main genes control the production of eumelanin (black/brown) and phaeomelanin (red/orange). The sex-linked orange gene on the X chromosome explains why most orange cats are male and most calico cats are female. Other genes control pattern, dilution, and white spotting."),
    ("What is a chimera cat?", "A chimera cat has two sets of DNA, resulting from the fusion of two fertilized eggs. This can cause striking asymmetry in coat color, with one side of the face or body being a different color than the other. Chimeras are rare and not a specific breed. Venus the cat is a famous example."),
    ("What is feline dwarfism?", "Feline dwarfism is a genetic condition that results in abnormally small cats. Breeds like the Munchkin have achondroplasia (short-legged dwarfism). Dwarf cats may have health issues including spinal problems, joint issues, and organ abnormalities. Not all small cats have dwarfism; some are simply petite."),
    ("What is a hypoallergenic cat?", "No cat breed is 100% hypoallergenic, but some breeds produce less Fel d1 protein, the main allergen. Siberian, Balinese, Cornish Rex, Devon Rex, Sphynx, and Russian Blue cats are often better tolerated by allergy sufferers. Individual reactions vary, so spend time with a breed before adopting."),
]

# ──────────────────────────────────────────────────────────────────────────────
# GENERATE FILES
# ──────────────────────────────────────────────────────────────────────────────

def generate_qa_text(pairs: list, repetitions: int = 5) -> str:
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

    # Clean up the narrative
    narrative_clean = "\n\n".join(p.strip() for p in CAT_FACTS.strip().split("\n\n") if p.strip())

    # Generate Q&A text
    qa_text = generate_qa_text(QA_PAIRS, repetitions=5)

    # Write combined corpus
    combined = generate_narrative_with_qa(narrative_clean, qa_text)
    with open("data/cat_all.txt", "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"OK - data/cat_all.txt created: {len(combined):,} characters (combined corpus)")

    total_narrative = len(narrative_clean)
    total_qa = len(qa_text)
    print(f"\nDataset statistics:")
    print(f"   Narrative text:     {total_narrative:>8,} characters")
    print(f"   Q&A text:           {total_qa:>8,} characters")
    print(f"   Total:              {len(combined):>8,} characters")
    print(f"   Unique Q&A pairs:   {len(QA_PAIRS):>8,}")
    print(f"   Total exchanges:    {len(QA_PAIRS) * 5:>8,}")
    print(f"\nReady for training!")
    print(f'  python -m metis train --dataset data/cat_all.txt --preset small --tokenizer cl100k_base --iters 5000 --use-moe --num-experts 4 --moe-top-k 2 --n-kv-heads 2 --use-qk-norm --use-ema')


if __name__ == "__main__":
    main()
