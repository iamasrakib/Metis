#!/usr/bin/env python
"""
Generate a comprehensive knowledge dataset about cows (cattle).

Creates in data/:
  - cow_all.txt — Combined corpus of all cow knowledge (narrative + Q&A)

Run:  python data/generate_cow_dataset.py
"""

import os

# ──────────────────────────────────────────────────────────────────────────────
# COW FACTS — Narrative knowledge base
# ──────────────────────────────────────────────────────────────────────────────

COW_FACTS = """
Cows are large domesticated herbivorous mammals raised for their milk, meat, and hides. The domestic cow belongs to the species Bos taurus, and the humped cattle of Asia and Africa belong to Bos indicus, also called zebu. Cattle are members of the family Bovidae and are the most numerous large livestock animal in the world. There are more than one billion cattle on Earth.

All domestic cattle descended from the aurochs, a massive wild ox that roamed Europe, Asia, and North Africa. The aurochs, Bos primigenius, was domesticated around 10,500 years ago in the Fertile Crescent of the Middle East. A second domestication event occurred later in the Indus Valley region of South Asia. The last wild aurochs died in Poland in 1627.

Cattle are ruminants, meaning they have a specialized four-chambered stomach. The four chambers are the rumen, the reticulum, the omasum, and the abomasum. Ruminants can digest plant material that most other animals cannot. The rumen is the largest chamber and can hold up to 40 gallons in an adult cow.

Cows chew their cud, which is the partially digested plant material that returns from the rumen to the mouth. A cow spends six to eight hours a day chewing her cud. Chewing the cud grinds the plant matter down further and mixes it with saliva, which contains bicarbonate to buffer the rumen. Cows have no upper front teeth, only a hard pad called a dental pad.

Cows are herbivores that eat grass, hay, silage, grains, and other plant matter. A dairy cow can eat more than 100 pounds of feed and drink 30 to 50 gallons of water in a single day. Cattle spend six to eleven hours a day eating and another eight hours chewing their cud.

The average cow produces about 6 to 8 gallons of milk per day. A high-producing dairy cow can produce over 10 gallons a day. A cow's lactation period lasts about 305 days. In a single lactation, a Holstein cow can produce more than 2,000 gallons of milk. Cows must give birth to a calf each year to keep producing milk.

Cattle are highly social herd animals. They form strong social bonds and establish a dominance hierarchy within the herd. Cows recognize each other and can remember up to 50 or more individual herd mates. They also recognize individual humans and can distinguish faces.

Cows are very intelligent animals. They can learn to navigate mazes, solve problems, and remember locations of food for a long time. Cows show emotional states such as excitement, calm, fear, and distress. A cow's heart rate rises when she is separated from her herd, which shows they form strong attachments.

Cows communicate with each other through vocalizations called moos, body language, and scent. A cow's moo can vary in pitch and volume to signal hunger, stress, greeting, or a call to her calf. Each cow has a distinctive voice, and calves can recognize their mother's call within days of birth.

Cows see in color, but not as well as humans. They are dichromatic, meaning they have two types of color receptors, and they cannot see red well. This is why matadors use red capes mostly for the audience, not the bull. Cows have nearly panoramic vision of about 330 degrees, but they have a blind spot directly behind them.

A cow has excellent hearing and can hear both lower and higher frequencies than humans. Cattle can also smell very well. A bull can detect a cow in heat from more than a mile away using scent.

Cows do not have upper front teeth. Instead they have a tough dental pad on the upper jaw. They use their lower teeth and tongue to grip and tear grass. Calves are born with baby teeth and develop their full set of 32 adult teeth by about five years of age. A cow's age can be estimated by her teeth.

Cattle have four legs and walk on two hoofed toes per foot. Their hooves grow continuously and are trimmed by farmers every few months to prevent lameness. A cow has no collar bone, and her spine is very flexible. Cows can run at speeds up to 25 miles per hour over short distances.

Cattle have a massive digestive system that produces large amounts of methane gas. Methane is released mostly through burping and to a smaller degree through flatulence. Methane from cattle is a significant source of greenhouse gas. Scientists are developing feed additives that can reduce methane emissions from cattle.

A cow's normal body temperature is about 101.5 degrees Fahrenheit. Her heart beats about 60 to 70 times per minute. A cow breathes 10 to 30 times per minute. Cows sweat very little, so they rely on panting and seeking shade to cool down in hot weather.

Cattle are usually born as a single calf. Twins occur in about 2 to 5 percent of births. The gestation period of a cow is about 283 days, which is close to nine months. A newborn calf weighs 60 to 100 pounds and can stand and walk within an hour of birth.

Calves drink their mother's milk, called colostrum for the first few days. Colostrum is rich in antibodies that protect the newborn calf from disease. After about two months, calves begin eating solid food and are weaned at four to six months of age.

Cows reach sexual maturity at about 12 to 18 months of age. A cow's estrus, or heat, cycle lasts about 21 days. A cow is in heat for about 18 hours, during which she is receptive to the bull. Most commercial farmers use artificial insemination to breed their cows.

A heifer is a female cow that has not yet had a calf. A cow is a female that has given birth at least once. A bull is an adult intact male. A steer is a castrated male, and an ox is a castrated male trained to pull loads. A calf is a young cow of either sex under one year old.

Dairy farmers separate most calves from their mothers shortly after birth. The calves are fed milk replacer or their mother's colostrum. This allows the farmer to collect the milk for human consumption. The calf is kept separate to prevent suckling and to monitor its health closely.

Holstein cows are the most recognizable dairy breed, with their black and white patchy coats. Holsteins are the highest-producing dairy breed, averaging about 23,000 pounds of milk per year. A Holstein cow weighs 1,500 to 1,600 pounds. Holsteins originated in the Netherlands.

Jersey cows are a small dairy breed with light brown coats. Jersey milk is extremely rich, with the highest butterfat and protein content of all dairy breeds. Jerseys weigh only 800 to 1,200 pounds but produce large amounts of high-quality milk for their size. Jerseys are known for being gentle and easy to handle.

The Guernsey cow is a brown and white dairy breed from the island of Guernsey. Guernsey milk has a golden color because of its high beta-carotene content. The Ayrshire is a red and white dairy breed from Scotland, known for its hardiness and strong constitution. The Brown Swiss is one of the oldest dairy breeds and produces milk that is excellent for cheese making.

Beef cattle are raised specifically for their meat. The most popular beef breed is the Angus, which is naturally polled, meaning it has no horns. Angus beef is known for its high-quality marbling, the white streaks of fat within the meat. Hereford cattle are red with white faces and are known for their hardiness and good temperament.

Wagyu is a Japanese beef breed famous for producing heavily marbled, tender, and expensive beef. True Wagyu beef comes from cattle such as the Japanese Black and Japanese Brown breeds. Kobe beef is a type of Wagyu from the Hyogo prefecture of Japan. Wagyu cattle are raised with great care to produce their distinctive meat.

The Brahman is a zebu breed of cattle from India, known for the hump on its shoulders and its loose, drooping skin. Brahman cattle tolerate heat very well and are resistant to many tropical diseases. They are often crossbred with European breeds to create heat-tolerant hybrids such as the Santa Gertrudis and the Brangus.

Highland cattle are a hardy Scottish breed with long, shaggy hair and large horns. Highland cattle can survive in harsh, cold, and rainy conditions where other cattle struggle. Their long hair protects them from the weather. They produce lean, well-marbled beef.

Cattle are raised for many products besides milk and meat. Cowhide is used to make leather goods such as shoes, jackets, and bags. Tallow, the rendered fat of cattle, is used in soap, candles, and cosmetics. Gelatin, used in food and medicine, is made from cattle bones and connective tissue. Rennet from a calf's stomach is traditionally used to make cheese.

Beef is divided into major cuts including chuck, rib, loin, round, flank, brisket, and shank. The most expensive cuts come from the rib and loin, such as ribeye and filet mignon. Ground beef is made from trimmings and less tender cuts. The United States is the world's largest beef producer and consumer.

Grass-fed beef comes from cattle that graze on pasture for their entire lives. Grain-fed cattle are finished on a diet of corn, barley, or other grains for the last few months before slaughter. Grass-fed beef is leaner and has a different flavor profile than grain-fed beef. Both methods have supporters and critics.

Cattle ranching is a major industry worldwide. The largest cattle ranches are in Australia, South America, and the western United States. The term cowboy comes from the men who herded cattle on horseback in the American West. Cattle drives in the 1800s moved millions of Longhorn cattle from Texas to railroad towns in Kansas.

India has the largest cattle population in the world, with more than 300 million cattle. Cows are considered sacred in Hinduism, and slaughtering a cow is illegal in many Indian states. In India, cows are used for milk, plowing, and as a source of fertilizer, but not for beef in most regions. The zebu breeds of India are adapted to the hot climate.

The cow has been a sacred animal for thousands of years in India. The cow is associated with the goddess Kamadhenu, who is believed to grant all wishes. In Hindu tradition, protecting and caring for cows is considered a virtuous act. Many Hindu festivals include rituals that honor the cow.

Cattle appear in the myths and religions of many cultures. In ancient Egypt, the sky goddess Nut was sometimes depicted as a cow. The Apis bull was a sacred bull worshiped in Memphis, Egypt. In Norse mythology, the giant cow Audumbla nourished the first giants with her milk. In Greek myth, Zeus carried the princess Europa away in the form of a white bull.

Beef is forbidden in several religions. In Hinduism, cows are sacred and beef is largely prohibited. In Judaism and Islam, cattle must be slaughtered according to specific religious rules to be considered kosher or halal. Observant Jews do not eat dairy and meat together, following the biblical law.

Mad cow disease, officially called bovine spongiform encephalopathy or BSE, is a fatal disease of cattle. BSE attacks the brain and nervous system of cattle. The disease emerged in the United Kingdom in the 1980s and was linked to feeding cattle meat-and-bone meal from infected animals. BSE can be transmitted to humans who eat contaminated nervous tissue, causing a disease called variant Creutzfeldt-Jakob disease.

Foot-and-mouth disease is a highly contagious viral disease of cattle, pigs, sheep, and other cloven-hoofed animals. It causes fever and painful blisters in the mouth and on the feet. Foot-and-mouth disease spreads very easily and can cause severe economic losses. It is not the same as hand, foot, and mouth disease in humans.

Mastitis is an inflammation of the udder, usually caused by bacterial infection. Mastitis is one of the most costly diseases in the dairy industry. It reduces milk yield and can contaminate milk with bacteria. Farmers prevent mastitis by keeping the udder clean, using good milking hygiene, and treating infected cows with antibiotics.

Bloat is a life-threatening condition where gas builds up in the rumen and cannot be released. Bloat can be caused by eating too much lush legume pasture such as alfalfa or clover. The swelling stomach can press on the lungs and cause death within hours if not treated. Farmers manage bloat with careful grazing management and anti-foaming agents.

Ketosis is a metabolic disease of high-producing dairy cows that occurs shortly after calving. When a cow cannot eat enough to meet the energy demand of milk production, her body breaks down fat, producing ketones. Ketosis causes weight loss, reduced milk yield, and a sweet smell on the cow's breath. It is treated with glucose and energy-dense feed.

Milk fever, also called parturient paresis, is a disease of calcium deficiency that occurs around the time of calving. When a cow begins producing large amounts of colostrum and milk, calcium leaves her blood very quickly. Milk fever causes weakness, paralysis, and even death if untreated. It is prevented by careful feeding before calving.

Cattle are vaccinated against many diseases to keep them healthy. Common cattle vaccines protect against infectious bovine rhinotracheitis, bovine viral diarrhea, leptospirosis, and clostridial diseases. Calves receive their first vaccinations in the first months of life. Good herd health programs prevent most major cattle diseases.

Cattle are identified on farms with ear tags, tattoos, and increasingly with electronic identification. Ear tags have unique numbers that track each animal's history. In many countries, cattle must be registered in a national database to trace their movements. This tracing system helps control disease outbreaks quickly.

Farmers care for cattle by providing clean water, nutritious feed, shelter, and veterinary care. Cattle need dry bedding to prevent disease and injury. Hoof care is important because overgrown hooves cause lameness and pain. Routine health checks help catch problems early. Treating animals humanely is both an ethical duty and good for production.

Hay is dried grass or legume that is stored and fed to cattle when pasture is not available. Silage is fermented green plant material, usually corn or grass, that is stored in airtight conditions. Pasture is the natural grazing land that cattle eat directly. Concentrates are high-energy grains such as corn, barley, and soy that supplement the forage.

A feedlot is a large, fenced facility where cattle are fattened on a high-energy grain diet before slaughter. Feedlots allow producers to raise many cattle in a small area. The beef industry has been criticized for the environmental impact and animal welfare concerns of feedlots. Grass-fed beef operations avoid feedlots entirely.

Cows produce milk in their udders. Milk leaves the udder through four teats, one for each quarter of the udder. Each quarter is a separate milk-producing gland. Cows are milked two to three times a day on most farms, using either milking machines or by hand. Milking takes about five to seven minutes per cow with a machine.

Cow's milk is rich in calcium, protein, and many vitamins. Whole milk contains about 87 percent water and 13 percent solids. Milk fat, also called butterfat, gives milk its creaminess. Milk is pasteurized to kill harmful bacteria by heating it and then cooling it quickly. Pasteurization is named after the scientist Louis Pasteur.

Dairy products made from cow's milk include cheese, butter, yogurt, cream, and ice cream. Cheese is made by coagulating milk with rennet or acid and aging the curds. Butter is made by churning cream until the fat separates. Yogurt is made by fermenting milk with live bacteria cultures. Each type of cheese has its own production method and flavor.

Lactose is the natural sugar found in milk. Many people cannot digest lactose, a condition called lactose intolerance. Lactose-intolerant people lack enough of the enzyme lactase in their gut. Lactose-free milk and dairy alternatives are available for people who cannot digest milk sugar.

Butter is made by churning cream until the butterfat clumps together and separates from the buttermilk. Butter is about 80 percent fat. Butter has been made and eaten by humans for thousands of years. Ghee, a clarified butter used in Indian cooking, is made by simmering butter to remove the water and milk solids.

Cheese has been made for over 7,000 years. Hard cheeses such as cheddar and parmesan are aged for months or years. Soft cheeses such as brie and camembert have a creamy interior and a soft rind. Fresh cheeses such as mozzarella and ricotta are not aged. There are thousands of varieties of cheese made from cow's milk.

Yogurt is made by fermenting milk with bacteria that convert lactose into lactic acid. The lactic acid gives yogurt its tangy flavor and thick texture. Yogurt contains live cultures that can be beneficial for digestive health. Greek yogurt is strained to remove whey, making it thicker and higher in protein.

Cattle contribute to climate change through the methane they release when digesting food. Methane is a greenhouse gas more than 25 times more potent than carbon dioxide over 100 years. Ruminant livestock are responsible for a significant share of global methane emissions. Improving feed efficiency and using additives such as seaweed can reduce methane production.

The environmental impact of cattle is a subject of intense debate. Some studies argue that beef production has a large carbon footprint compared to plant foods. Other researchers point out that well-managed grazing can store carbon in the soil and support biodiversity. Sustainable ranching practices aim to balance livestock production with environmental health.

Cattle ranching has been linked to deforestation in the Amazon rainforest. Large areas of forest have been cleared to create pasture for cattle. Deforestation reduces biodiversity and releases stored carbon. Consumers and companies are increasingly demanding beef that is produced without deforestation.

The cattle industry is a major part of the world economy. The global cattle population is about 1.5 billion animals. Brazil has the largest commercial cattle herd in the world, followed by India, China, and the United States. The dairy industry provides livelihoods for millions of farming families worldwide.

Cows have a very strong sense of smell and can detect scents from up to six miles away. They can smell water underground and sense rain from a distance. Cows also use smell to identify their calves and other members of the herd. This remarkable sense of smell guides much of their behavior.

Cows have excellent memories and can remember people and places for years. They learn routines quickly, such as the milking schedule, and can become stressed if it changes. Cows can remember how to solve tasks even after long gaps. A cow's memory and intelligence are often underestimated.

Cows synchronize their behavior. When one cow lies down to rest, others tend to lie down too. When one cow starts grazing, the herd begins grazing together. This synchronization is part of their herd survival instinct. Cattle feel safer grazing in groups, with some members watching for danger.

Cows show a preference for lying on soft, dry bedding and will choose comfortable areas to rest. Cows spend 10 to 12 hours a day lying down, which is important for their health and rumination. Lying on hard or wet surfaces increases the risk of lameness and injury. Good cow comfort improves milk production and well-being.

Cattle can see a full range of colors except red. This means the red cape used by matadors is actually invisible as red to the bull; it is the motion of the cape that attracts the bull's attention. Cattle are also sensitive to motion and can detect movement up to half a mile away.

A group of cattle is called a herd. A female that has given birth is a cow. A young female that has not calved is a heifer. An intact adult male is a bull. A castrated male is a steer. A castrated male trained as a draft animal is an ox. A baby is a calf. The collective term for a large group of cattle is sometimes a drove.

The oldest cow ever recorded was a Holstein-Dairy Shorthorn cross named Big Bertha, who lived to the age of 49. She produced 39 calves during her lifetime in Ireland. The heaviest cow ever recorded weighed over 3,000 pounds. The heaviest bull ever recorded, a Holstein named Darth Vader, weighed about 3,600 pounds.

A cow has four stomachs in one, which is why farmers say cows have four stomachs. The rumen is a fermentation vat filled with billions of microbes that break down cellulose. The microbes in a single cow's rumen could fill a large drum. This microbial digestion is what allows cows to eat grass that humans cannot digest.

Cows do not have a true upper front row of teeth. Their lower incisors meet a hard, tough pad on the upper jaw. To graze, a cow wraps her tongue around the grass and slices it with her lower teeth against the pad. This is why grazing cattle leave grass cut cleanly rather than pulled up by the roots.

Bulls can weigh over 2,000 pounds and are powerful and potentially dangerous animals. Mature bulls are aggressive during breeding season. Bull rings and nose leads are used to control bulls. Experienced farmers treat bulls with great respect and caution. A bull's strength is legendary and must never be underestimated.

Cattle are prey animals and have a strong fight-or-flight response. Sudden movements and loud noises frighten cattle. Calm, quiet handling reduces stress in cattle. Stressed cattle release cortisol, which can lower meat quality and reduce milk production. Low-stress handling is a core principle of modern cattle farming.

Cows are ruminants that swallow their food without chewing it fully. Later they bring it back up as a cud and chew it more thoroughly. This process is called rumination. Rumination allows cattle to eat quickly in the open and chew safely later in a protected place. It is an adaptation that helped wild cattle survive predators.

In the United States, more than 90 percent of dairy cows are Holsteins or Holstein crosses. The United States is the world's largest producer of beef and the second-largest producer of milk. The state of Wisconsin is called America's Dairyland. Texas is the state with the most cattle in the United States.

Cattle were brought to the Americas by Spanish explorers in the 1500s. These cattle escaped and became the wild herds of Texas, including the famous Texas Longhorn. Later, British and other European settlers brought their own breeds. The cattle industry shaped the history and culture of the American West.

The Texas Longhorn is descended from the cattle brought by the Spanish. Longhorns are known for their extremely long horns, which can span over six feet from tip to tip. Longhorns are hardy, disease-resistant, and can survive on sparse pasture. They nearly went extinct in the 1900s but were saved by conservation efforts.

Cattle are used for draft work in many parts of the world, pulling plows, carts, and wagons. An ox is a castrated male trained for this work. Oxen are strong, patient, and reliable work animals. In some countries cattle are still essential for farming and transportation where machinery is not available.

The dung of cattle is a valuable resource on farms. Cow dung is used as natural fertilizer because it returns nutrients to the soil. Dried cow dung is used as fuel in many parts of the world. In India, cow dung is used in traditional building materials and rituals. Cow dung also feeds dung beetles and supports soil ecosystems.

Cow urine is used in some traditional medicines and agricultural practices, particularly in India. In Ayurveda, cow products including milk, ghee, urine, and dung are called panchagavya. Panchagavya is used in traditional rituals and remedies. These practices are part of the cultural reverence for the cow in India.

Some dairy farmers keep cows on pasture year-round, while others keep cows indoors in freestall barns. Freestall barns give each cow a comfortable, individual resting stall with soft bedding. Cows in well-managed barns can be very healthy and comfortable. Both systems can work well if managed with attention to cow comfort and hygiene.

The average dairy cow is milked until her milk production declines, then she is dried off for about two months before calving again. This dry period lets the udder rest and regenerate. Dairy cows typically produce milk for about five to seven years before their production declines. A good dairy cow can produce milk for many lactations.

Bovine milk is one of the most nutritionally complete foods for humans. It contains protein, fat, carbohydrates, vitamins A, D, and B12, calcium, phosphorus, and potassium. Milk is especially important for growing children because of its calcium and protein. Nutritionists recommend dairy as part of a balanced diet.

Chocolate milk is milk mixed with cocoa and sweetener. It is popular with children and athletes for its taste and energy content. Flavored milks use sugar and flavorings that make them higher in calories than plain milk. Some schools limit flavored milk because of its added sugar.

Skim milk and low-fat milk have most of the fat removed. Whole milk contains about 3.25 percent fat. Removing the fat also removes the fat-soluble vitamins A and D, which are then often added back. Many people prefer low-fat dairy for health reasons, while others prefer the richer taste of whole milk.

Ice cream is a frozen dessert made from milk, cream, sugar, and flavorings. To be called ice cream in the United States, a product must contain at least 10 percent milk fat. Gelato is an Italian style ice cream with less fat and air. Ice cream is one of the most popular dairy products in the world.

The world's largest consumers of dairy are European countries, the United States, and India. Per-capita milk consumption is highest in Europe and North America. India is the largest producer of milk in the world, ahead of the United States, China, and Brazil. Global milk production exceeds 900 million tons per year.

Cattle breeding has produced over 1,000 distinct cattle breeds worldwide. Breeds are grouped into dairy breeds, beef breeds, and dual-purpose breeds that provide both milk and meat. Some breeds are adapted to tropical climates, while others thrive in cold regions. Breed choice is one of the most important decisions a cattle farmer makes.

Dual-purpose breeds such as the Simmental and the Shorthorn provide both milk and meat. Simmental cattle originated in Switzerland and are one of the oldest and most widespread cattle breeds. The Milking Shorthorn and the Red Poll are also dual-purpose breeds. Dual-purpose breeds give farmers flexibility to sell either milk or meat as markets change.

The zebu, or Bos indicus, cattle are distinguished from European cattle by their hump and drooping ears. Zebu cattle tolerate heat and resist parasites better than European breeds. Zebu cattle originated in South Asia and spread to Africa and the Americas. Their adaptability makes them essential for cattle production in tropical regions.

Crossbreeding combines the strengths of different cattle breeds. Brahman and Angus crosses produce beef cattle that tolerate heat while still producing high-quality meat. Holstein and Jersey crosses combine high milk volume with rich milk. Crossbreeding can improve disease resistance, fertility, and production. Many commercial herds use planned crossbreeding programs.

The Charolais is a white French beef breed known for its rapid growth and heavy muscling. The Limousin is another French beef breed valued for its lean, high-yielding meat. The Gelbvieh is a German dual-purpose breed. The Belgian Blue has a genetic mutation that causes extreme muscle development, producing very lean meat. Continental European breeds have transformed beef production.

Modern dairy farms use computers to track each cow's milk production, health, and reproduction. Robotic milking systems let cows choose when to be milked. Sensors monitor cow activity to detect heat and illness early. Dairy technology continues to improve efficiency and cow welfare. A modern dairy cow is a highly productive and carefully managed animal.

Cows are remarkable for their ability to convert grass, which humans cannot eat, into nutritious milk and meat. This makes cattle valuable on land that cannot grow crops. Grazing cattle help maintain grassland ecosystems when managed well. The relationship between cattle and grasslands is thousands of years old.

Calves are born with their eyes open and can stand within minutes. A healthy calf is on its feet and nursing within an hour of birth. Calves gain weight very quickly, often doubling their birth weight in the first month. Calves are curious and playful, running and jumping in what farmers call the zoomies.

A cow's tail is used to swat flies and other insects. The tail is made of a long bone covered by a tuft of hair. Flies and parasites are a major nuisance and health problem for cattle. Many farmers use fly control programs to protect their herds.

Dehorning cattle is common in the dairy industry to prevent injuries to other animals and people. Polled breeds such as Angus naturally have no horns. Dehorning is done at a young age when it causes the least stress. Some countries restrict or ban dehorning without pain relief. Disbudding calves early is considered less painful than removing horns later.

Cattle are ruminants along with sheep, goats, deer, and giraffes. Ruminants are defined by their four-chambered stomachs and cud-chewing behavior. The word ruminant comes from the Latin word ruminare, which means to chew again. Ruminants play a vital role in converting grass into protein for human food.

Beef is a rich source of high-quality protein, iron, zinc, and B vitamins. Red meat such as beef provides complete protein containing all essential amino acids. Health organizations recommend eating red meat in moderation as part of a balanced diet. Lean beef can be part of a heart-healthy diet when eaten in sensible portions.

The way cattle are raised affects the environment, the animals, and the quality of the product. Pasture-raised cattle graze on grass, while conventional cattle may spend time in feedlots. Organic beef comes from cattle raised without antibiotics or growth hormones. Consumers can choose beef that matches their values and budget.

Cow-calf operations are the first stage of beef production, where cows give birth and raise calves. Weaned calves are sent to a backgrounding operation, where they grow on forage. Finally, the cattle are moved to feedlots for finishing on grain. Each stage has its own management practices and challenges.

Veal is the meat of young calves, usually dairy bull calves. Veal production has been controversial because of past welfare concerns about confining calves. Modern veal producers provide calves with more space, social contact, and iron in their diet. Veal is a pale, tender meat that is popular in European cuisine.

The dairy industry and the beef industry are connected. Dairy farms produce bull calves that are raised for beef, often called dairy-beef. Some beef cattle are also bred from dairy cows. Dairy cows that are no longer productive are culled and sold for beef. The two industries support each other in many countries.

Cattle have a long history with humans, dating back more than 10,000 years. They have provided milk, meat, hides, and draft power. Cattle have shaped landscapes, economies, and cultures around the world. The relationship between humans and cattle is one of the oldest and most important partnerships in history.

Farmers practice good stewardship by rotating cattle through pastures to prevent overgrazing. Rotational grazing lets grass recover between grazing periods. Well-managed pastures support healthy soil, clean water, and wildlife. Regenerative grazing is a modern movement that uses cattle to build healthy soil and capture carbon.

The cow is a truly remarkable animal. She turns sunlight into grass, grass into milk and meat, and supports millions of human livelihoods. From sacred temple cow to modern high-tech dairy farm, cattle have walked alongside human civilization for ten millennia. Understanding and caring for cattle well is a responsibility that farmers take seriously.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Q&A CORPUS — Structured knowledge in chat format
# ──────────────────────────────────────────────────────────────────────────────

QA_PAIRS = [
    # ── Basic Cow Facts ──
    ("What is a cow?", "A cow is a large domesticated herbivorous mammal raised for milk, meat, and hides. The domestic cow belongs to the species Bos taurus, and the humped cattle of Asia and Africa belong to Bos indicus. Cows are members of the family Bovidae."),
    ("What is the scientific name of a cow?", "The scientific name of the domestic cow is Bos taurus. The humped cattle of Asia and Africa are Bos indicus, also called zebu. All domestic cattle are descended from the wild aurochs, Bos primigenius."),
    ("Where did cows come from?", "All domestic cattle descended from the aurochs, a massive wild ox that lived in Europe, Asia, and North Africa. The aurochs was domesticated about 10,500 years ago in the Fertile Crescent. A second domestication happened in the Indus Valley. The last wild aurochs died in Poland in 1627."),
    ("How long have cows been domesticated?", "Cattle have been domesticated for about 10,500 years. The first domestication occurred in the Fertile Crescent of the Middle East around 8,500 BC. A later domestication event happened in the Indus Valley of South Asia. Cattle are among the oldest domesticated farm animals."),
    ("How many cows are there in the world?", "There are more than one billion cattle in the world. The global cattle population is estimated at about 1.5 billion animals. India has the largest cattle population, followed by Brazil and China. Cattle are the most numerous large livestock animals on Earth."),
    ("Are cows ruminants?", "Yes, cows are ruminants. Ruminants are animals with a specialized four-chambered stomach that can digest tough plant material. Cattle, sheep, goats, deer, and giraffes are all ruminants. The word ruminant comes from the Latin word meaning to chew again."),

    # ── Cow Anatomy ──
    ("How many stomachs does a cow have?", "A cow has one stomach with four chambers: the rumen, the reticulum, the omasum, and the abomasum. The rumen is the largest chamber and can hold up to 40 gallons. The microbes in the rumen break down cellulose, which is why cows can digest grass."),
    ("What is the rumen?", "The rumen is the largest chamber of a cow's stomach. It is a fermentation vat filled with billions of microbes that break down cellulose in plant material. The rumen can hold up to 40 gallons in an adult cow. This microbial digestion lets cows eat grass humans cannot digest."),
    ("Why do cows chew their cud?", "Cows chew their cud to grind down partially digested plant material from the rumen. Chewing the cud mixes it with saliva, which buffers the rumen. A cow spends six to eight hours a day chewing her cud. This process is called rumination."),
    ("Do cows have upper teeth?", "No, cows do not have upper front teeth. They have a hard pad called the dental pad on their upper jaw. Cows use their lower teeth and tongue to grip and tear grass against this pad. This is why grazing cattle leave grass cut cleanly."),
    ("How many teeth do cows have?", "Cows develop a full set of 32 adult teeth by about five years of age. They have no upper incisors or canines, only a dental pad. A cow's age can be estimated by examining her teeth. Calves are born with baby teeth."),
    ("How many stomachs does a farmer mean when they say a cow has four stomachs?", "When farmers say a cow has four stomachs, they mean the four chambers of her single stomach: the rumen, reticulum, omasum, and abomasum. Each chamber has a different function in digestion. The rumen ferments, the omasum absorbs water, and the abomasum is the true stomach."),
    ("How fast can a cow run?", "Cows can run at speeds up to 25 miles per hour over short distances. They have a flexible spine and no collarbone, which helps them move. Despite their size, cattle are surprisingly agile. They usually walk at a leisurely pace but can sprint when frightened."),
    ("What is a cow's normal temperature?", "A cow's normal body temperature is about 101.5 degrees Fahrenheit, which is higher than a human's. A cow's heart beats about 60 to 70 times per minute. She breathes 10 to 30 times per minute. Cows rely on panting and shade to cool down because they sweat very little."),
    ("Can cows see color?", "Yes, cows can see color, but they are dichromatic, meaning they have only two types of color receptors. They cannot see red well; it appears dull to them. This is why matadors use red capes mainly for the audience. The motion of the cape, not its color, attracts the bull."),
    ("What is the field of vision of a cow?", "Cattle have nearly panoramic vision of about 330 degrees because their eyes are on the sides of their heads. They have a blind spot directly behind them. Cows are very sensitive to motion and can detect movement up to half a mile away."),
    ("How good is a cow's sense of smell?", "A cow's sense of smell is very powerful. Cows can detect scents from up to six miles away and can smell water underground. They use smell to identify their calves and herd mates. A bull can detect a cow in heat from more than a mile away."),
    ("How good is a cow's hearing?", "Cattle have excellent hearing and can hear both lower and higher frequencies than humans. A cow's ears are mobile and can rotate to locate sounds. Cows are sensitive to the calls of their calves. Loud noises frighten cattle and cause stress."),
    ("Why do cows have a tail?", "A cow's tail is used to swat flies and other insects. The tail is a long bone covered by a tuft of hair at the end. Flies and parasites are a major nuisance and health problem for cattle. Farmers use fly control programs to protect their herds."),
    ("Do cows sweat?", "Cows sweat very little because they have few sweat glands. Instead, they rely on panting, seeking shade, and licking their coats to cool down. Their coat can help reflect heat. Heat stress is a serious problem for cattle in hot weather."),
    ("How much does a cow weigh?", "A dairy cow such as a Holstein weighs 1,500 to 1,600 pounds. A Jersey cow weighs only 800 to 1,200 pounds. Beef bulls can weigh over 2,000 pounds. The heaviest cow ever recorded weighed over 3,000 pounds."),

    # ── Cow Behavior ──
    ("Are cows social animals?", "Yes, cattle are highly social herd animals. They form strong social bonds and establish a dominance hierarchy. Cows recognize and remember up to 50 or more individual herd mates. They also recognize individual humans and can become stressed when separated from the herd."),
    ("Are cows intelligent?", "Yes, cows are very intelligent animals. They can learn to navigate mazes, solve problems, and remember locations of food for a long time. Cows show emotions such as excitement, calm, fear, and distress. Their memory and intelligence are often underestimated."),
    ("Why do cows moo?", "Cows moo to communicate with each other and with humans. A cow's moo varies in pitch and volume to signal hunger, stress, greeting, or a call to her calf. Each cow has a distinctive voice. Calves can recognize their mother's call within days of birth."),
    ("Do cows have best friends?", "Cows form strong bonds with specific herd mates, sometimes called best friends. They prefer to rest and graze near their preferred companions. When separated from their friends, cows show signs of stress. Pairing cows with their friends can even improve milk production."),
    ("Can cows recognize people?", "Yes, cows can recognize individual humans and distinguish faces. They remember people who treated them well or badly. Cows can learn routines, such as milking schedules, and become stressed if the routine changes. Their memory for people and places lasts for years."),
    ("How much do cows sleep?", "Cows sleep only about four hours a day, and only lightly. They spend 10 to 12 hours a day lying down, but much of that time is spent chewing their cud rather than sleeping. Cows do most of their grazing in the morning and evening. Cattle rest and ruminate lying down in groups."),
    ("How much time do cows spend eating?", "Cattle spend six to eleven hours a day eating and another eight hours chewing their cud. A dairy cow can eat more than 100 pounds of feed in a day. Cows also drink 30 to 50 gallons of water daily. Grazing and ruminating take up most of a cow's day."),
    ("Do cows get stressed?", "Yes, cows experience stress from separation, fear, pain, and changes in routine. Stressed cattle release cortisol, which can lower milk production and meat quality. Sudden movements and loud noises frighten cattle. Low-stress handling is a core principle of modern cattle farming."),
    ("Why do cows lie down?", "Cows lie down to rest, sleep, and chew their cud. Lying down is important for a cow's health because standing for too long causes stress on her hooves. Cows prefer soft, dry bedding to lie on. A cow lying down comfortably is usually a sign of good welfare."),
    ("Do cows synchronize their behavior?", "Yes, cows synchronize their behavior. When one cow lies down, the others tend to lie down too, and they graze together as a group. This synchronization is a herd survival instinct. Cattle feel safer grazing in groups with some members watching for danger."),

    # ── Cow Breeds ──
    ("What is the most popular dairy breed?", "The Holstein is the most popular and recognizable dairy breed, with its black and white patchy coat. Holsteins are the highest-producing dairy breed, averaging about 23,000 pounds of milk per year. More than 90 percent of dairy cows in the United States are Holsteins or Holstein crosses."),
    ("What is a Holstein cow?", "A Holstein is a black and white dairy cow from the Netherlands, known for producing the most milk of any breed. Holsteins average about 23,000 pounds of milk per year and weigh 1,500 to 1,600 pounds. They are the most common dairy breed in the world."),
    ("What is a Jersey cow?", "A Jersey is a small dairy breed with a light brown coat, known for milk with the highest butterfat and protein content. Jerseys weigh only 800 to 1,200 pounds. They are gentle and easy to handle. Jersey milk is prized for making cheese and butter."),
    ("What is the best beef cattle breed?", "The Angus is the most popular beef breed, known for high-quality marbling and naturally polled (hornless) cattle. Angus beef is prized for its tenderness and flavor. Hereford cattle are also popular for their hardiness and good temperament. The best breed depends on the climate and market."),
    ("What is Wagyu beef?", "Wagyu is a Japanese beef breed famous for producing heavily marbled, tender, and expensive beef. Kobe beef is a type of Wagyu from Hyogo prefecture in Japan. Wagyu cattle are raised with great care to produce their distinctive meat. The heavy marbling gives Wagyu its buttery texture."),
    ("What is a Brahman cow?", "A Brahman is a zebu breed from India, recognized by its shoulder hump and drooping skin. Brahman cattle tolerate heat well and resist tropical diseases. They are often crossbred with European breeds to create heat-tolerant cattle such as the Santa Gertrudis and Brangus."),
    ("What is a Highland cow?", "A Highland cow is a hardy Scottish breed with long, shaggy hair and large horns. Highland cattle survive in harsh, cold, rainy conditions where other cattle struggle. Their long hair protects them from the weather. They produce lean, well-marbled beef."),
    ("What are zebu cattle?", "Zebu are humped cattle of the species Bos indicus from South Asia and Africa. Zebu have a hump on their shoulders and tolerate heat and parasites better than European cattle. They spread from South Asia to Africa and the Americas. Zebu are essential for cattle production in the tropics."),
    ("How many cattle breeds are there?", "There are more than 1,000 distinct cattle breeds in the world. Breeds are grouped into dairy breeds, beef breeds, and dual-purpose breeds. Some breeds are adapted to tropical climates and others to cold regions. Breed choice is one of the most important decisions a cattle farmer makes."),
    ("What is a dual-purpose cattle breed?", "A dual-purpose breed provides both milk and meat. Examples include the Simmental, the Shorthorn, and the Red Poll. Simmental cattle originated in Switzerland and are among the oldest and most widespread breeds. Dual-purpose breeds let farmers sell either milk or meat as markets change."),
    ("What is a Charolais cow?", "A Charolais is a white French beef breed known for rapid growth and heavy muscling. Charolais cattle produce lean, high-yielding meat. They are one of the major continental European beef breeds. Charolais have been used worldwide to improve beef production."),
    ("What is a Simmental cow?", "The Simmental is a large dual-purpose breed from Switzerland, providing both milk and meat. It is one of the oldest and most widespread cattle breeds in the world. Simmentals are known for rapid growth and good temperament. They come in many coat colors including red and white."),
    ("What is a Texas Longhorn?", "The Texas Longhorn descends from cattle brought to the Americas by Spanish explorers. Longhorns are known for their extremely long horns, which can span over six feet. They are hardy, disease-resistant, and survive on sparse pasture. They nearly went extinct but were saved by conservation efforts."),
    ("What is a Belgian Blue?", "The Belgian Blue is a beef breed with a genetic mutation that causes extreme muscle development. The mutation reduces the production of myostatin, the protein that limits muscle growth. Belgian Blues produce very lean, heavily muscled meat. The breed is sometimes called the double-muscled cow."),

    # ── Milk & Dairy ──
    ("How much milk does a cow produce?", "The average cow produces about 6 to 8 gallons of milk per day. A high-producing dairy cow can produce over 10 gallons a day. In a single lactation of about 305 days, a Holstein can produce more than 2,000 gallons. A cow must give birth to a calf each year to keep producing milk."),
    ("How long is a cow's lactation?", "A cow's lactation period lasts about 305 days. After lactation, the cow is dried off for about two months before calving again. Dairy cows typically produce milk for five to seven years. A good dairy cow can have many lactations in her lifetime."),
    ("How many teats does a cow have?", "A cow has four teats, one for each quarter of her udder. Each quarter is a separate milk-producing gland. Cows are milked two to three times a day on most farms. Milking takes about five to seven minutes per cow with a machine."),
    ("What is in cow's milk?", "Cow's milk is about 87 percent water and 13 percent solids. It contains protein, fat, carbohydrates, vitamins A, D, and B12, calcium, phosphorus, and potassium. Milk fat, also called butterfat, gives milk its creaminess. Milk is one of the most nutritionally complete foods."),
    ("What is pasteurization?", "Pasteurization is heating milk to kill harmful bacteria, then cooling it quickly. It is named after scientist Louis Pasteur. Pasteurized milk is safe to drink and lasts longer. Raw milk is unpasteurized and carries a higher risk of foodborne illness."),
    ("What is lactose?", "Lactose is the natural sugar found in milk. Many people cannot digest it, a condition called lactose intolerance. Lactose-intolerant people lack enough of the enzyme lactase. Lactose-free milk and dairy alternatives are available for them."),
    ("What is butter made from?", "Butter is made by churning cream until the butterfat clumps together and separates from the buttermilk. Butter is about 80 percent fat. Ghee, a clarified butter used in Indian cooking, is made by simmering butter to remove water and milk solids."),
    ("How is cheese made?", "Cheese is made by coagulating milk with rennet or acid, then draining the whey and aging the curds. Rennet from a calf's stomach is traditionally used to set the curd. Hard cheeses are aged for months or years, while fresh cheeses are not aged. There are thousands of cheese varieties."),
    ("How is yogurt made?", "Yogurt is made by fermenting milk with live bacteria that convert lactose into lactic acid. The lactic acid gives yogurt its tangy flavor and thick texture. Greek yogurt is strained to remove whey, making it thicker and higher in protein. Yogurt contains live cultures beneficial for digestion."),
    ("What is whole milk?", "Whole milk contains about 3.25 percent milk fat. Skim milk and low-fat milk have most of the fat removed. Removing fat also removes the fat-soluble vitamins A and D, which are often added back. Many people prefer low-fat dairy for health, while others prefer the taste of whole milk."),
    ("What is colostrum?", "Colostrum is the first milk a cow produces after calving, rich in antibodies that protect the newborn calf from disease. Calves must receive colostrum within the first hours of life. Colostrum is thick and yellowish compared to normal milk. It is essential for the calf's immune system."),
    ("What dairy products come from cows?", "Dairy products made from cow's milk include cheese, butter, yogurt, cream, ice cream, and condensed milk. Chocolate milk is milk mixed with cocoa and sweetener. Each type of cheese has its own production method and flavor. Dairy is a major part of diets worldwide."),
    ("Which country produces the most milk?", "India is the largest producer of milk in the world, ahead of the United States, China, and Brazil. Global milk production exceeds 900 million tons per year. Per-capita milk consumption is highest in Europe and North America. India's large cattle population drives its milk production."),
    ("What is ice cream made from?", "Ice cream is a frozen dessert made from milk, cream, sugar, and flavorings. To be called ice cream in the United States, it must contain at least 10 percent milk fat. Gelato is an Italian style with less fat and air. Ice cream is one of the most popular dairy products."),
    ("Why do calves drink colostrum?", "Calves drink colostrum, the first milk after birth, because it is packed with antibodies. These antibodies protect the newborn calf from disease before its own immune system matures. Colostrum must be given within the first hours of life to be absorbed. Without it, calves are very vulnerable to infection."),

    # ── Beef ──
    ("What is beef?", "Beef is the meat of cattle. It is divided into major cuts including chuck, rib, loin, round, flank, brisket, and shank. The most expensive cuts come from the rib and loin, such as ribeye and filet mignon. The United States is the world's largest beef producer and consumer."),
    ("What is marbling?", "Marbling is the white streaks of fat within beef muscle. More marbling means more tender, flavorful, and expensive meat. Angus beef is known for its high-quality marbling. Wagyu beef has extreme marbling that gives it a buttery texture."),
    ("What is grass-fed beef?", "Grass-fed beef comes from cattle that graze on pasture for their entire lives. Grass-fed beef is leaner and has a different flavor than grain-fed beef. It is often marketed as more natural and sustainable. Some grass-fed operations avoid feedlots entirely."),
    ("What is grain-fed beef?", "Grain-fed beef comes from cattle finished on a diet of corn, barley, or other grains for the last months before slaughter. The grain diet promotes faster growth and heavier marbling. Most conventional beef in the United States is grain-finished. Grass-fed and grain-fed beef have different taste and fat content."),
    ("What is a feedlot?", "A feedlot is a large, fenced facility where cattle are fattened on a high-energy grain diet before slaughter. Feedlots allow producers to raise many cattle in a small area. They have been criticized for environmental and welfare concerns. Grass-fed operations avoid feedlots entirely."),
    ("What is veal?", "Veal is the meat of young calves, usually dairy bull calves. Veal production was controversial because of past welfare concerns. Modern veal producers provide more space, social contact, and iron. Veal is a pale, tender meat popular in European cuisine."),
    ("What is the most expensive cut of beef?", "The most expensive cuts of beef come from the rib and loin, including ribeye, filet mignon, and T-bone. Japanese Wagyu, especially Kobe beef, is among the most expensive beef in the world. High marbling, tenderness, and scarcity drive beef prices. Prices vary widely by market and quality grade."),
    ("What is a cow-calf operation?", "A cow-calf operation is the first stage of beef production, where cows give birth and raise calves. Weaned calves then go to a backgrounding operation to grow on forage. Finally, cattle are moved to feedlots for grain finishing. Each stage has its own management practices."),
    ("Is beef nutritious?", "Beef is a rich source of high-quality protein, iron, zinc, and B vitamins. Red meat provides complete protein with all essential amino acids. Health organizations recommend eating red meat in moderation. Lean beef can be part of a heart-healthy diet in sensible portions."),
    ("What is dairy-beef?", "Dairy-beef is beef produced from cattle raised on dairy farms, usually bull calves. Dairy farms produce bull calves that are raised for beef. Some beef cattle are also bred from dairy cows. Cows that are no longer productive are culled and sold for beef."),

    # ── Farming & Care ──
    ("What do cows eat?", "Cows are herbivores that eat grass, hay, silage, grains, and other plant matter. Hay is dried grass or legume stored for when pasture is unavailable. Silage is fermented green plant material. Concentrates are high-energy grains that supplement the forage."),
    ("What is hay?", "Hay is dried grass or legume that is stored and fed to cattle when pasture is not available. It is cut, dried, and baled. Good hay is nutritious and free of mold. Hay is the most common stored feed for cattle."),
    ("What is silage?", "Silage is fermented green plant material, usually corn or grass, stored in airtight conditions. The fermentation preserves the nutrients and prevents spoilage. Silage is a high-moisture feed commonly used in dairy farming. It is stored in silos, bunkers, or wrapped bales."),
    ("How much water does a cow drink?", "A dairy cow drinks 30 to 50 gallons of water in a single day. Water is essential for milk production and digestion. Cows need clean, fresh water available at all times. Water intake increases in hot weather and during lactation."),
    ("How do farmers care for cows?", "Farmers care for cows by providing clean water, nutritious feed, shelter, and veterinary care. Cows need dry bedding to prevent disease and injury. Hoof care prevents lameness, and routine health checks catch problems early. Treating animals humanely is both ethical and good for production."),
    ("What is a freestall barn?", "A freestall barn is an indoor dairy housing system where each cow has a comfortable, individual resting stall with soft bedding. Freestall barns protect cows from weather and keep them clean. Well-managed barns support healthy, comfortable cows. Some farms instead keep cows on pasture year-round."),
    ("Why do farmers trim cow hooves?", "Farmers trim cow hooves because they grow continuously and cause lameness if overgrown. Lameness is painful and reduces milk production. Hooves are trimmed every few months, often by a professional hoof trimmer. Regular hoof care is essential for cow welfare."),
    ("Why do farmers dehorn cattle?", "Farmers dehorn cattle to prevent injuries to other animals and people. Polled breeds such as Angus naturally have no horns. Dehorning is done at a young age to minimize stress. Some countries restrict dehorning without pain relief. Disbudding young calves is considered less painful."),
    ("How are cattle identified?", "Cattle are identified with ear tags, tattoos, and increasingly with electronic identification. Ear tags have unique numbers that track each animal's history. Many countries require cattle to be registered in a national database. Tracing systems help control disease outbreaks quickly."),
    ("What is rotational grazing?", "Rotational grazing is moving cattle between pastures so grass can recover between grazing periods. It prevents overgrazing and keeps pastures healthy. Well-managed pastures support healthy soil, clean water, and wildlife. Regenerative grazing uses cattle to build soil and capture carbon."),
    ("Why is cow comfort important?", "Cow comfort is important because comfortable cows are healthier and produce more milk. Cows prefer soft, dry bedding and spend 10 to 12 hours a day lying down. Lying on hard or wet surfaces causes lameness and injury. Good cow comfort is both ethical and profitable."),
    ("What are common cattle vaccines?", "Common cattle vaccines protect against infectious bovine rhinotracheitis, bovine viral diarrhea, leptospirosis, and clostridial diseases. Calves receive their first vaccinations in the first months of life. Vaccination is part of a good herd health program. Most major cattle diseases are preventable."),

    # ── Cow Health & Disease ──
    ("What is mad cow disease?", "Mad cow disease, officially bovine spongiform encephalopathy or BSE, is a fatal disease of cattle that attacks the brain and nervous system. It emerged in the UK in the 1980s, linked to feeding cattle meat-and-bone meal from infected animals. BSE can transmit to humans through contaminated nervous tissue, causing variant Creutzfeldt-Jakob disease."),
    ("What is foot-and-mouth disease?", "Foot-and-mouth disease is a highly contagious viral disease of cattle, pigs, sheep, and other cloven-hoofed animals. It causes fever and painful blisters in the mouth and feet. The disease spreads easily and causes severe economic losses. It is not the same as hand, foot, and mouth disease in humans."),
    ("What is mastitis?", "Mastitis is an inflammation of the udder, usually caused by bacterial infection. It is one of the most costly diseases in the dairy industry because it reduces milk yield and contaminates milk. Farmers prevent mastitis with clean udders and good milking hygiene. Infected cows are treated with antibiotics."),
    ("What is bloat in cattle?", "Bloat is a life-threatening condition where gas builds up in the rumen and cannot escape. It can be caused by eating too much lush legume pasture such as alfalfa or clover. The swelling stomach presses on the lungs and can cause death within hours. Farmers manage bloat with careful grazing and anti-foaming agents."),
    ("What is ketosis in cows?", "Ketosis is a metabolic disease of high-producing dairy cows that occurs after calving. When a cow cannot eat enough to meet milk production energy demands, her body breaks down fat and produces ketones. Ketosis causes weight loss, reduced milk yield, and a sweet breath smell. It is treated with glucose and energy-dense feed."),
    ("What is milk fever?", "Milk fever, or parturient paresis, is a calcium deficiency disease that occurs around calving. When a cow begins producing colostrum and milk, calcium leaves her blood very quickly. Milk fever causes weakness, paralysis, and even death if untreated. It is prevented by careful feeding before calving."),
    ("What is Johne's disease?", "Johne's disease is a chronic, incurable bacterial infection of the intestines in cattle. It causes severe diarrhea, weight loss, and eventually death. It is caused by Mycobacterium avium subspecies paratuberculosis. The disease has a long incubation period and is hard to control once introduced."),
    ("Why do cows get lameness?", "Lameness in cows is usually caused by overgrown hooves, hoof infections, or injury. It is painful and reduces milk production and fertility. Wet, dirty conditions increase the risk of hoof problems. Regular hoof trimming and clean bedding prevent most lameness."),
    ("Can humans get diseases from cattle?", "Yes, some cattle diseases can spread to humans; these are called zoonotic diseases. Examples include E. coli infections from contaminated milk or meat, brucellosis, and variant Creutzfeldt-Jakob disease from BSE. Proper pasteurization, cooking, and hygiene prevent most zoonotic spread."),
    ("What is brucellosis?", "Brucellosis is a bacterial disease of cattle that causes abortion and reduced fertility. It is zoonotic, meaning it can spread to humans, causing undulant fever. It is controlled through testing, vaccination, and slaughter of infected herds. Many countries have eradicated the disease."),

    # ── Reproduction ──
    ("How long is a cow pregnant?", "A cow's gestation period is about 283 days, which is close to nine months. Most cows carry a single calf. Twins occur in about 2 to 5 percent of births. A newborn calf weighs 60 to 100 pounds."),
    ("How long is a cow's heat cycle?", "A cow's estrus, or heat, cycle lasts about 21 days. A cow is in heat for about 18 hours, during which she is receptive to the bull. Farmers detect heat by watching for mounting behavior and increased activity. Most commercial farmers use artificial insemination to breed cows."),
    ("When does a cow first come into heat?", "Heifers reach sexual maturity at about 12 to 18 months of age. The first heat, called puberty, depends on breed, weight, and nutrition. Farmers usually breed heifers when they reach about 55 to 65 percent of their mature weight. Breeding too early can harm the young cow's growth."),
    ("What is artificial insemination?", "Artificial insemination, or AI, is breeding cattle by placing semen into a cow's reproductive tract without natural mating. AI allows farmers to use semen from superior bulls anywhere in the world. It improves genetics and reduces the risk of disease. Most dairy cows are bred by artificial insemination."),
    ("What is a heifer?", "A heifer is a young female cow that has not yet given birth to a calf. Once a heifer has her first calf, she is called a cow. Heifers are bred at about 12 to 18 months of age. They calve for the first time at about two years old."),
    ("What is the difference between a cow, bull, steer, and ox?", "A cow is a female that has given birth at least once. A bull is an adult intact male. A steer is a castrated male. An ox is a castrated male trained to pull loads. A heifer is a female that has not calved, and a calf is a young cow of either sex under one year old."),
    ("How soon after birth can a calf stand?", "A healthy calf can stand and walk within an hour of birth. Calves are born with their eyes open. They find their mother's udder and nurse within the first hours. Early movement and nursing are vital for the calf to receive colostrum."),
    ("When are calves weaned?", "Calves drink their mother's milk for the first weeks of life. They begin eating solid food at about two months. Calves are typically weaned at four to six months of age. Weaning is gradual to reduce stress."),
    ("What is a calf's first milk called?", "A calf's first milk is called colostrum, produced in the first days after birth. Colostrum is thick, yellowish, and rich in antibodies. It protects the newborn calf from disease. Calves must receive colostrum within the first hours of life."),
    ("Why are dairy calves separated from their mothers?", "Dairy farmers separate most calves from their mothers shortly after birth to collect the mother's milk for humans. The calves are fed milk replacer or colostrum separately. This lets the farmer monitor the calf's health and prevent suckling. The separation is controversial and some farms raise cow-calf together."),

    # ── History & Culture ──
    ("Why are cows sacred in India?", "Cows are sacred in India because of their central role in Hinduism. The cow is associated with the goddess Kamadhenu, who grants all wishes. Protecting and caring for cows is considered a virtuous act. Slaughtering a cow is illegal in many Indian states."),
    ("What is Kamadhenu?", "Kamadhenu is the divine cow of Hindu mythology, said to grant all wishes. She is sometimes called the cow of plenty. Kamadhenu represents abundance and the nourishing power of the cow. She is worshiped as a mother goddess in Hindu tradition."),
    ("What role did cows play in ancient Egypt?", "In ancient Egypt, the sky goddess Nut was sometimes depicted as a cow. The Apis bull was a sacred bull worshiped in Memphis, symbolizing strength and fertility. Bulls were often mummified and buried in grand ceremonies. Cattle were central to Egyptian religion and agriculture."),
    ("What is the Apis bull?", "The Apis bull was a sacred bull worshiped in ancient Memphis, Egypt. It was believed to be an incarnation of the god Ptah. When an Apis bull died, it was mummified and buried with great ceremony. The Apis bull symbolized strength, fertility, and the life-giving power of cattle."),
    ("What is the story of Europa and the bull?", "In Greek mythology, Zeus, king of the gods, transformed himself into a beautiful white bull to carry away the princess Europa. He carried her across the sea to the island of Crete. The continent of Europe is named after her. The story shows the importance of the bull in ancient Mediterranean culture."),
    ("What is Audumbla?", "Audumbla is a giant primordial cow in Norse mythology. Her milk nourished the first giant, Ymir. Audumbla is one of the earliest creatures in Norse creation myths. She appears in the Prose Edda, the great collection of Norse myths."),
    ("Why is beef forbidden in some religions?", "In Hinduism, cows are sacred and beef is largely prohibited. In Judaism and Islam, cattle must be slaughtered by specific religious rules to be kosher or halal. Observant Jews do not eat dairy and meat together. Food laws in these religions reflect deep cultural and religious values."),
    ("How did cattle come to America?", "Cattle were brought to the Americas by Spanish explorers in the 1500s. These cattle escaped and became the wild herds of Texas, including the Texas Longhorn. British and other European settlers later brought their own breeds. Cattle shaped the history and culture of the American West."),
    ("What is a cattle drive?", "A cattle drive is the long-distance herding of cattle on horseback. In the 1800s, cowboys drove millions of Longhorn cattle from Texas to railroad towns in Kansas. The drives moved cattle to markets and railheads. Cattle drives became a defining image of the American West."),
    ("What is a cowboy?", "A cowboy is a cattle herder who works on horseback, especially in the American West. Cowboys drove cattle across long distances in the 1800s. Their clothing, tools, and lifestyle became legendary. The cowboy culture of the American West grew out of Mexican vaquero traditions."),
    ("Why do matadors use red capes?", "Matadors use red capes mainly for the audience, not the bull, because bulls cannot see red well. Bulls are attracted by the motion of the cape. Cattle are dichromatic and see red as a dull color. The myth that red enrages bulls is one of the most common misconceptions about cattle."),
    ("What is panchagavya?", "Panchagavya is a traditional Indian mixture of five cow products: milk, curd, ghee, urine, and dung. It is used in Ayurvedic medicine and traditional rituals. Panchagavya reflects the cultural reverence for the cow in India. The practice is thousands of years old."),

    # ── Environment & Economy ──
    ("Why do cows produce methane?", "Cows produce methane through enteric fermentation, the digestion of food by microbes in the rumen. Methane is released mostly through burping and, to a lesser degree, flatulence. Methane is a potent greenhouse gas. Scientists are developing feed additives to reduce methane from cattle."),
    ("How much methane do cows produce?", "A single cow can produce about 200 to 400 pounds of methane per year. Methane is more than 25 times more potent than carbon dioxide as a greenhouse gas. Ruminant livestock contribute a significant share of global methane emissions. Feed additives and better feed efficiency can reduce these emissions."),
    ("What is enteric fermentation?", "Enteric fermentation is the digestive process by which microbes in a ruminant's stomach break down feed and produce methane. The methane is released through burping. It is the main source of cattle's greenhouse gas emissions. Reducing enteric fermentation is a major goal of climate research."),
    ("Is beef bad for the environment?", "Beef production has a larger carbon footprint than most plant foods because cattle emit methane and require much land. However, well-managed grazing can store carbon in soil and support biodiversity. The environmental impact depends heavily on how the cattle are raised. Sustainable ranching aims to balance production with environmental health."),
    ("How does cattle ranching affect the Amazon?", "Cattle ranching is a major cause of deforestation in the Amazon rainforest. Large areas of forest are cleared to create pasture. Deforestation reduces biodiversity and releases stored carbon. Consumers and companies increasingly demand beef produced without deforestation."),
    ("What is the largest cattle herd in the world?", "India has the largest cattle population in the world, with more than 300 million cattle. Brazil has the largest commercial cattle herd, followed by India, China, and the United States. The global cattle population is about 1.5 billion animals. The cattle industry is a major part of the world economy."),
    ("What country produces the most beef?", "The United States is the world's largest beef producer and consumer. Brazil is the second-largest producer and the largest beef exporter. India is one of the largest beef producers but consumes little because of religious beliefs. Global beef production is dominated by a handful of large countries."),
    ("What is regenerative ranching?", "Regenerative ranching is a movement that uses cattle to build healthy soil and capture carbon. Practices include rotational grazing and careful pasture management. Well-managed grazing can improve soil fertility, water quality, and biodiversity. Regenerative ranching aims to make beef production part of the climate solution."),
    ("How are cattle used besides for meat and milk?", "Cattle provide hides for leather, tallow for soap and candles, and gelatin for food and medicine. Cattle are used for draft work, pulling plows and carts. Cow dung is used as fertilizer and fuel. Cattle contribute to many industries beyond meat and dairy."),
    ("What is cowhide used for?", "Cowhide is the skin of cattle, used to make leather goods such as shoes, jackets, belts, and bags. The leather-making process preserves the hide. Cattle hides are a valuable byproduct of the meat industry. Leather is durable, versatile, and used worldwide."),

    # ── Fun Facts ──
    ("What is a group of cows called?", "A group of cattle is called a herd. A large group being moved together is sometimes called a drove. Within a herd, cattle form a dominance hierarchy. Cattle are highly social and prefer to stay in their herd."),
    ("What is the oldest cow ever recorded?", "The oldest cow ever recorded was Big Bertha, a Holstein-Dairy Shorthorn cross from Ireland who lived to age 49. She produced 39 calves during her lifetime. Most cows live about 15 to 20 years. Big Bertha held two world records: oldest cow and most calves."),
    ("What is the heaviest cow ever recorded?", "The heaviest cow ever recorded weighed over 3,000 pounds. The heaviest bull ever recorded, a Holstein named Darth Vader, weighed about 3,600 pounds. Most dairy cows weigh 1,200 to 1,600 pounds. Beef bulls can exceed 2,000 pounds."),
    ("Can cows remember people?", "Yes, cows have excellent memories and can remember people and places for years. They recognize individual humans and distinguish faces. Cows remember how to solve tasks even after long gaps. Their memory and intelligence are often underestimated."),
    ("Do cows get lonely?", "Yes, cows can become distressed when isolated from their herd. They form strong social bonds and prefer to be near their companions. A cow's heart rate rises when she is separated from the herd. Cattle are social animals that need companionship to thrive."),
    ("Can cows run fast?", "Cows can run at speeds up to 25 miles per hour over short distances. They have a flexible spine and no collarbone. Despite their large size, cattle are surprisingly agile. They usually walk slowly but can sprint when alarmed."),
    ("Do cows have four stomachs?", "Strictly, a cow has one stomach with four chambers: the rumen, reticulum, omasum, and abomasum. Farmers say cows have four stomachs to describe these chambers. Each chamber plays a different role in digestion. The rumen is the largest and does most of the fermentation."),
    ("Why do cows have spots?", "Cow spots are determined by genetics and vary by breed. Holsteins are black and white, Guernseys are brown and white, and Jerseys are solid light brown. The pattern of spots is unique to each animal, like a fingerprint. Spots have no special purpose, though some believe they help with camouflage."),
    ("Can cows swim?", "Yes, cows are strong swimmers and can cross rivers when needed. During floods, cattle have been known to swim long distances to safety. Their large lung capacity helps them float. Ranchers in flood-prone areas sometimes see cattle swimming to high ground."),
    ("What does the cow say?", "The sound a cow makes is called a moo. Cows moo to communicate hunger, stress, greeting, and to call their calves. Each cow has a distinctive voice. Calves recognize their mother's call within days of birth."),

    # ── Care & Handling ──
    ("How do you handle cows safely?", "Handle cattle calmly and quietly because sudden movements and loud noises frighten them. Use low-stress handling techniques and give cattle space. Bulls can be dangerous and must be treated with great respect. Understanding cattle behavior keeps both people and animals safe."),
    ("Why are cattle calm in herds?", "Cattle feel safer in herds because there is safety in numbers. In a group, some animals watch for danger while others graze. Predators in the wild find it harder to single out one animal. This instinct keeps cattle calm in groups."),
    ("How do cattle show stress?", "Cattle show stress through increased heart rate, restlessness, bawling, and changes in eating. Stressed cattle release cortisol, which lowers milk production and meat quality. Loud noise, crowding, and isolation cause stress. Calm, consistent handling reduces stress."),
    ("Do cows like being petted?", "Many cows enjoy gentle handling and can become very tame. Dairy cows that are handled regularly often seek human contact. However, cattle can be startled by sudden touch. Each cow has a different personality and comfort level with people."),
    ("Can cows be trained?", "Yes, cows are intelligent and can be trained using positive reinforcement. They learn routines, navigate obstacles, and can be taught to enter milking parlors and follow handlers. Calves learn quickly when trained gently. Their intelligence makes them responsive to consistent handling."),
    ("Why do cows line up to be milked?", "Cows learn the milking routine and line up at the parlor because they are creatures of habit and often relieved when their full udder is milked. The relief of milk pressure is rewarding. Cows are very routine-oriented and become stressed if the schedule changes."),
    ("What is low-stress cattle handling?", "Low-stress cattle handling uses calm movement, understanding of cattle flight zones, and positive experience to move cattle without fear. It avoids shouting, prodding, and crowding. Low-stress handling improves animal welfare and meat quality. It is a core principle of modern cattle farming."),

    # ── Milk Production Details ──
    ("Why must a cow have a calf to produce milk?", "A cow produces milk in response to giving birth. The birth triggers hormonal changes that start lactation. A cow must be bred and calve about once a year to keep producing milk. Milk production declines after about ten months and the cow is dried off."),
    ("What is a milk replacer?", "Milk replacer is a formula fed to calves instead of their mother's milk. It contains milk proteins, fats, vitamins, and minerals. It is mixed with water and fed from a bottle or bucket. Milk replacer lets dairy farms raise calves while collecting the mother's milk for sale."),
    ("How does a milking machine work?", "A milking machine uses vacuum to gently draw milk from the cow's teats into a pipeline. The milk flows to a bulk tank where it is cooled and stored. Milking takes about five to seven minutes per cow. Modern robotic systems let cows be milked without human handling."),
    ("What is robotic milking?", "Robotic milking systems let cows choose when to be milked, around the clock. A robot cleans the udder, attaches the milking cups, and monitors milk quality. Cows are often eager to visit the robot for feed rewards. Robotic milking improves cow comfort and reduces labor."),
    ("How many times a day are cows milked?", "Most dairy cows are milked two to three times a day. Milking frequency affects milk production, with more milkings yielding more milk. Milking is done at regular times because cows are creatures of habit. Each milking takes about five to seven minutes per cow."),
    ("What happens to milk after it leaves the farm?", "After leaving the farm, milk is transported by tanker truck to a processing plant. The plant tests, pasteurizes, and homogenizes the milk. It is then packaged and delivered to stores. Some milk is processed into cheese, butter, yogurt, and other dairy products."),
    ("What is homogenization?", "Homogenization is a process that breaks up milk fat globules so the cream does not separate and rise to the top. It forces the milk through tiny openings under high pressure. Homogenized milk has a uniform texture. Most commercial milk is homogenized."),

    # ── More Cow Facts ──
    ("Why do farmers rotate crops and cattle?", "Farmers rotate crops and cattle to keep soil healthy. Cattle manure fertilizes the soil, and pasture rest between grazing lets grass recover. Rotational grazing prevents overgrazing and supports biodiversity. Well-managed land cycles between crops, pasture, and livestock."),
    ("What is the difference between dairy and beef cattle?", "Dairy cattle are bred to produce large quantities of milk, while beef cattle are bred for meat production. Dairy breeds include Holstein and Jersey; beef breeds include Angus and Hereford. Dual-purpose breeds provide both. The two industries have very different management practices."),
    ("How long do cows live?", "Most cows live 15 to 20 years. Dairy cows are usually productive for five to seven years before their milk production declines. Beef cattle are usually raised to about 18 to 24 months before slaughter. The oldest cow on record lived to age 49."),
    ("Can cows recognize their calves?", "Yes, cows recognize their calves by sight, sound, and smell. A cow and her calf bond within hours of birth. The cow will call to her calf with a distinctive lowing, and the calf recognizes her mother's call. The bond is strongest in the first months of life."),
    ("Why are cattle considered sacred in some cultures?", "Cattle are considered sacred in cultures where they provide essential food and labor. In India, the cow is sacred in Hinduism for its milk, dung, and role in agriculture. In ancient Egypt, the Apis bull was worshiped. Cattle have been vital to human survival for thousands of years."),
    ("What is the economic importance of cattle?", "Cattle are economically vital worldwide. The dairy and beef industries employ millions of people. Global milk production exceeds 900 million tons per year. Cattle provide food, leather, and draft power. The cattle industry is a cornerstone of the global food system."),
    ("How has dairy farming changed over time?", "Dairy farming has changed from small family farms with hand milking to large, high-tech operations. Modern farms use milking machines, robots, and computer tracking of each cow. Milk yields per cow have risen dramatically through breeding and nutrition. Animal welfare and sustainability are growing priorities."),
    ("Why do some people avoid dairy?", "Some people avoid dairy because of lactose intolerance, milk allergies, veganism, or environmental and ethical concerns. Lactose intolerance is the inability to digest milk sugar. Many plant-based milk alternatives such as soy and oat milk are available. Dairy consumption is a personal choice."),
]


# ──────────────────────────────────────────────────────────────────────────────
# GENERATE FILES
# ──────────────────────────────────────────────────────────────────────────────

def generate_qa_text(pairs: list, repetitions: int = 4) -> str:
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
    narrative_clean = "\n\n".join(p.strip() for p in COW_FACTS.strip().split("\n\n") if p.strip())

    # Generate Q&A text
    qa_text = generate_qa_text(QA_PAIRS, repetitions=4)

    # Write combined corpus
    combined = generate_narrative_with_qa(narrative_clean, qa_text)
    with open("data/cow_all.txt", "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"OK - data/cow_all.txt created: {len(combined):,} characters (combined corpus)")

    total_narrative = len(narrative_clean)
    total_qa = len(qa_text)
    print(f"\nDataset statistics:")
    print(f"   Narrative text:     {total_narrative:>8,} characters")
    print(f"   Q&A text:           {total_qa:>8,} characters")
    print(f"   Total:              {len(combined):>8,} characters")
    print(f"   Unique Q&A pairs:   {len(QA_PAIRS):>8,}")
    print(f"   Total exchanges:    {len(QA_PAIRS) * 4:>8,}")
    print(f"\nReady for training!")
    print(f"  metis train --dataset data/cow_all.txt")


if __name__ == "__main__":
    main()
