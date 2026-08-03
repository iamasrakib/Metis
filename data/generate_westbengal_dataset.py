#!/usr/bin/env python
"""Generate a comprehensive West Bengal knowledge dataset for training."""

import os
import sys

DATASET = """
West Bengal is a state in the eastern region of India. It is the fourth most populous
state in India with a population of over 91 million people. The state capital is
Kolkata, which is the third largest city in India. West Bengal covers an area of
88,752 square kilometers and shares borders with Bangladesh to the east, Nepal and
Bhutan to the north, and the Indian states of Odisha, Jharkhand, Bihar, Sikkim, and
Assam.

The geography of West Bengal is remarkably diverse. The state stretches from the
Himalayan peaks in the north to the Bay of Bengal in the south. The northern part
includes the Darjeeling Hills, which are famous for their tea plantations and
breathtaking views of Kanchenjunga, the third highest mountain in the world at
8,586 meters. The Ganges River flows through the state, creating the vast Bengal
delta, which is the largest delta in the world. The Sundarbans, a UNESCO World
Heritage Site, is located in the southern part of the state and is home to the
Royal Bengal Tiger.

West Bengal has a rich and complex history. The region was part of the ancient
Magadha and Maurya empires. Bengal was a major center of the Indian independence
movement. The Bengal Renaissance of the 19th and early 20th centuries was a period
of extraordinary cultural, intellectual, and social reform. Key figures of this
movement include Ram Mohan Roy, who is considered the father of the Indian
Renaissance, and Ishwar Chandra Vidyasagar, who worked tirelessly for women's
education and the abolition of child marriage.

Kolkata, the capital city, was formerly known as Calcutta and served as the capital
of British India until 1911. The city is known as the Cultural Capital of India.
It is home to the Indian Museum, which is the oldest and largest museum in India,
founded in 1814. The Victoria Memorial, a magnificent white marble building, is
one of the most iconic landmarks of Kolkata. The Howrah Bridge, officially known
as Rabindra Setu, is one of the busiest bridges in the world, connecting Kolkata
with its twin city of Howrah across the Hooghly River.

The culture of West Bengal is deeply rooted in literature, music, art, and cinema.
Bengali literature has a glorious tradition dating back centuries. Bankim Chandra
Chatterjee wrote Vande Mataram, which became one of the national songs of India.
Rabindranath Tagore, born in Kolkata, was the first non-European to win the Nobel
Prize in Literature in 1913. He wrote Jana Gana Mana, which became the national
anthem of India. Another literary giant, Kazi Nazrul Islam, known as the Rebel
Poet, wrote extensively about social justice and revolution.

Bengali cinema, commonly known as Tollywood, has produced some of the finest films
in Indian cinema. Satyajit Ray, one of the greatest filmmakers in history, created
the Apu Trilogy, which is considered among the finest works of world cinema. His
film Pather Panchali won numerous international awards and put Indian cinema on the
global map. Other notable Bengali filmmakers include Ritwik Ghatak, Mrinal Sen,
and more recently, Rituparno Ghosh.

The music of West Bengal is incredibly diverse. Rabindra Sangeet, songs composed
by Rabindranath Tagore, forms an integral part of Bengali culture. Nazrul Geeti,
songs by Kazi Nazrul Islam, is another major musical tradition. Baul music, a
mystical folk tradition of Bengal, was recognized by UNESCO as a Masterpiece of
the Oral and Intangible Heritage of Humanity. The Bauls are wandering minstrels
who sing about the divine and the human experience.

West Bengal is famous for its festivals. Durga Puja is the biggest festival of
the state, celebrated with enormous enthusiasm in October. The festival honors
the goddess Durga and her victory over the demon Mahishasura. During Durga Puja,
elaborately crafted pandals (temporary structures) are set up across the state,
each competing to be the most artistic and innovative. The festival was inscribed
on UNESCO's Representative List of the Intangible Cultural Heritage of Humanity
in 2021. Kali Puja, celebrated in November, honors the goddess Kali. Eid is
also widely celebrated in the state due to its significant Muslim population.

The cuisine of West Bengal is renowned for its richness and variety. Bengali
cuisine is characterized by its use of mustard oil, panch phoron (a five-spice
blend), and a wide variety of fish preparations. Rice and fish are staple foods,
and the phrase macher jhol bhaater chaal (fish curry and rice) is synonymous with
Bengali identity. Famous dishes include macher jhol (fish curry), shorshe ilish
(hilsa fish in mustard sauce), kosha mangsho (slow-cooked mutton), cholar dal
(Bengal gram lentils), and rasgulla, a beloved sweet made from cottage cheese.
Sweets hold a special place in Bengali culture, with iconic desserts like
sandesh, mishti doi (sweetened yogurt), and pantua.

The economy of West Bengal is the sixth largest in India. Agriculture is a major
sector, with the state being one of the largest producers of rice, jute, and tea
in India. The tea gardens of Darjeeling produce some of the finest tea in the
world, with Darjeeling tea holding a Geographical Indication tag. The jute
industry, historically known as the Golden Fiber, has been a backbone of the
Bengal economy since the colonial era. Kolkata is a major commercial and financial
center, home to the Calcutta Stock Exchange and numerous corporate headquarters.

West Bengal has made significant progress in education. The state is home to some
of India's oldest and most prestigious educational institutions. The University
of Calcutta, established in 1857, is one of the first multidisciplinary Western-style
universities in Asia. Presidency University, formerly Hindu College, was founded
in 1817 and has produced numerous distinguished alumni including Rabindranath
Tagore, Subhas Chandra Bose, and Amartya Sen. Jadavpur University, Indian
Statistical Institute, and Indian Institute of Science Education and Research
Kolkata are other notable institutions.

The state has produced many notable leaders and thinkers. Subhas Chandra Bose,
known as Netaji, was a charismatic leader who founded the Indian National Army
to fight for India's independence. Chittaranjan Das, known as Deshbandhu, was
a prominent freedom fighter and founder of the Swaraj Party. Sri Aurobindo,
originally from Kolkata, was a philosopher, poet, and Indian nationalist who
later became a spiritual reformer. Amartya Sen, born in Santiniketan, won the
Nobel Prize in Economics in 1998 for his work on welfare economics and social
choice theory.

West Bengal has 23 districts as of 2024. The major districts include Kolkata,
Howrah, North 24 Parganas, South 24 Parganas, Hooghly, Nadia, Murshidabad,
Birbhum, Burdwan, Purba Medinipur, Paschim Medinipur, Jalpaiguri, Darjeeling,
Cooch Behar, Malda, and Siliguri. Each district has its own unique cultural
identity and contributes to the diverse tapestry of the state.

The Sundarbans, shared between India and Bangladesh, is the largest mangrove
forest in the world. It is a UNESCO World Heritage Site and a Biosphere Reserve.
The forest is home to the Royal Bengal Tiger, saltwater crocodiles, and numerous
other species. The Sundarbans is also a critical barrier against cyclones and
storm surges for the coastal communities of West Bengal.

Transportation in West Bengal is well-developed. Kolkata has one of the oldest
metro rail systems in India, the Kolkata Metro, which began operations in 1984.
The city also has an extensive network of trams, one of the few cities in India
to still operate tram services. Netaji Subhas Chandra Bose International
Airport in Kolkata is the primary air gateway to eastern India. The Howrah
Railway Station and Sealdah Railway Station are two of the busiest railway
stations in India.

The sports culture of West Bengal is dominated by football and cricket. Kolkata
is considered the Mecca of Indian football, with the Salt Lake Stadium being
one of the largest football stadiums in India. The city hosts the historic
Durand Cup, one of the oldest football tournaments in Asia. The Eden Gardens
cricket stadium in Kolkata is one of the largest cricket stadiums in the world
and has hosted numerous historic cricket matches. East Bengal FC and Mohun
Bagan are two of the oldest and most storied football clubs in India.

The flora and fauna of West Bengal are incredibly diverse. The state ranges
from the alpine forests of the Himalayas in the north to the tropical mangroves
of the Sundarbans in the south. The forests of Darjeeling and the Dooars are
home to elephants, leopards, red pandas, and Himalayan black bears. The Jaldapara
National Park in Alipurduar district is the second largest wildlife sanctuary
in North Bengal and is famous for its population of Indian one-horned rhinoceros.

Bengal has a strong tradition of art and craftsmanship. The Kalighat painting
tradition originated in the 19th century near the Kalighat temple in Kolkata.
These paintings depict everyday life and social themes with bold lines and vivid
colors. The Shantiniketan school of art, founded by Rabindranath Tagore,
encouraged a fusion of Indian and Western artistic traditions. Bengal is also
known for its terracotta temples, particularly in Bishnupur, where intricate
clay sculptures adorn the walls of temples built during the Malla dynasty.

The state has a vibrant textile industry. Bengal muslin, once famous worldwide
for its exquisite fineness, was a major export during the Mughal era. The
handloom industry produces beautiful sarees including the famous Baluchari
saree, Tant saree, and Jamdani saree. Kantha, a traditional form of embroidery,
is practiced widely by women in rural Bengal. The craft involves running
stitches to create intricate patterns on layers of old cloth.

West Bengal's contribution to science and technology is noteworthy. The Indian
Statistical Institute in Kolkata, founded by Prasanta Chandra Mahalanobis, is
a premier institution for research in statistics and related sciences. C.V. Raman,
who won the Nobel Prize in Physics in 1930 for his work on light scattering,
conducted much of his research at the Indian Association for the Cultivation
of Science in Kolkata. Meghnad Saha, an astrophysicist from Kolkata, developed
the Saha ionization equation which is fundamental to understanding stellar
spectra.

The rivers of West Bengal are central to its geography and culture. The Ganges
enters West Bengal from Bihar and flows southward to the Bay of Bengal. The
Hooghly River, a distributary of the Ganges, flows through Kolkata and is
considered sacred. The Padma River flows along the border with Bangladesh.
The Teesta, Mahananda, and Jalpaiguri rivers flow through the northern districts.
These rivers support agriculture, fishing, and transportation throughout the state.

The climate of West Bengal varies significantly from north to south. The northern
hill regions have a temperate climate with cool summers and cold winters. The
plains experience a tropical climate with hot summers, heavy monsoon rains,
and mild winters. The monsoon season from June to September brings heavy
rainfall, which is crucial for agriculture. Cyclones from the Bay of Bengal
frequently affect the southern coastal districts, causing significant damage
to life and property.

West Bengal has a rich tradition of theater and performing arts. Jatra, a form
of folk theater, is immensely popular in rural Bengal. It involves elaborate
performances with music, dance, and drama. Bengali theater has a long and
glorious history, with institutions like the Star Theatre and Minerva Theatre
playing important roles in the cultural life of Kolkata. The state also has
a vibrant dance tradition, with Chhau dance being a notable classical dance
form recognized by UNESCO.

The tea industry of Darjeeling is world-renowned. The first tea plantation in
Darjeeling was established in 1841 by Dr. Archibald Campbell. Today, there
are over 80 tea estates in the Darjeeling district, producing tea that is
prized worldwide for its unique muscatel flavor. The tea industry provides
employment to thousands of workers, many of whom are women of Nepali and
Bhutia descent. Darjeeling tea was the first Indian product to receive a
Geographical Indication tag.

The political history of West Bengal is significant in Indian politics. The
state was a stronghold of the Indian National Congress during the independence
movement. After independence, the Communist Party of India (Marxist) governed
the state for 34 consecutive years from 1977 to 2011, the longest democratically
elected communist government in the world. The left government implemented land
reforms and rural development programs that were studied worldwide. In 2011,
the All India Trinamool Congress, led by Mamata Banerjee, came to power.

Mamata Banerjee, the current Chief Minister, is one of the most prominent
political figures in India. She founded the All India Trinamool Congress in
1998 and is known for her grassroots political activism. She was awarded the
Guddi Gudda Award and is sometimes referred to as Didi, meaning elder sister
in Bengali. Her government has focused on infrastructure development, industrial
revival, and social welfare programs.

The educational landscape of West Bengal includes numerous schools and colleges
affiliated with boards like the West Bengal Board of Secondary Education and
the West Bengal Council of Higher Secondary Education. The state performs well
in national entrance examinations, with many students qualifying for prestigious
institutions like the Indian Institutes of Technology. The state government has
invested heavily in digital literacy and online education platforms.

Siliguri, located in the northern part of West Bengal, is known as the Gateway
to Northeast India. It is a major commercial hub and transportation center,
connecting the northeastern states with the rest of India. The city is also
a gateway to Nepal, Bhutan, and Tibet. NJP (New Jalpaiguri) railway station
is one of the most important railway junctions in eastern India.

The Sundarbans mangrove ecosystem supports a unique biodiversity. The forest
is crisscrossed by a network of tidal waterways, mudflats, and small islands.
The Royal Bengal Tiger population in the Sundarbans is estimated at around 100
individuals. These tigers are known for their ability to swim in the saline
waters of the delta. The Sundarbans is also home to the Ganges River dolphin,
Irrawaddy dolphins, and saltwater crocodiles.

Pottery and terracotta work is an ancient craft tradition in West Bengal. The
town of Bishnupur in Bankura district is famous for its terracotta temples,
built by the Malla kings between the 17th and 18th centuries. The temples
feature intricate terracotta panels depicting scenes from the Ramayana and
Mahabharata. The craft of Bankura horse terracotta has become an iconic symbol
of Bengali folk art.

The state's handicraft industry includes bell metal work from Bankura, wood
carving from Shantiniketan, and brass work from Bishnupur. Kantha embroidery,
a traditional craft practiced by women, involves running stitches to create
beautiful patterns on layers of cloth. Each Kantha tells a story, often
depicting scenes from rural life, mythology, and nature.

West Bengal's connection to the Indian independence movement is profound. The
Partition of Bengal in 1905 by Lord Curzon sparked massive protests and the
Swadeshi movement. The Jallianwala Bagh massacre in Amritsar in 1919 had
deep repercussions in Bengal. The Quit India Movement of 1942 saw widespread
participation from Bengalis. The Tebhaga movement of 1946 was a significant
peasant uprising in Bengal demanding fair share of crops.

The state is also known for its unique form of Bengali, which has two main
dialects: Chittagonian in the south and Rangpuri in the north. Standard
Bengali, based on the Nadia dialect, is the official language. Bengali is
the seventh most spoken language in the world, with over 230 million speakers.
The Bengali script evolved from the Brahmi script and is written from left
to right.

The film industry of Kolkata, known as Tollywood, produces around 150 films
annually. In recent years, Bengali cinema has experienced a renaissance with
critically acclaimed films that have won awards at international film festivals.
Directors like Goutam Ghose, Aparna Sen, and Srijit Mukherji have gained
recognition for their artistic and socially relevant films. The Kolkata
International Film Festival is one of the oldest film festivals in Asia.

Healthcare in West Bengal has improved significantly over the years. The state
has several major medical institutions, including the Medical College and
Hospital in Kolkata, one of the oldest medical colleges in Asia, established
in 1835. The state government has launched several health insurance schemes
to provide affordable healthcare to the rural population.

The IT sector in Kolkata has grown rapidly in recent years. The Salt Lake
Sector V and Rajarhat New Town areas have become major IT hubs, hosting
offices of companies like TCS, Infosys, Wipro, and Cognizant. The state
government has promoted IT investment through favorable policies and the
establishment of Special Economic Zones.

West Bengal's environmental challenges include air pollution in Kolkata,
flooding during the monsoon, and the impact of climate change on the
Sundarbans. The state government has initiated several programs to address
these issues, including the Kolkata Environmental Improvement Project and
mangrove restoration programs in the Sundarbans.

The state has a vibrant tradition of folk music and dance. Baul singers are
recognizable by their distinctive dress and instrument, the ektara. Their
songs explore themes of love, devotion, and the search for the divine.
Lalon Shah, the most famous Baul saint, composed thousands of songs that
continue to inspire people across Bengal. His shrine in Kushtia, Bangladesh,
attracts devotees from both India and Bangladesh.

The banking sector in Kolkata dates back to the colonial era. The Bank of
Bengal, established in 1806, was one of the first modern banks in India
and later merged to form the Imperial Bank of India, which eventually became
the State Bank of India. Today, Kolkata is an important financial center
in eastern India with numerous national and international banks having
their regional headquarters in the city.

The state's infrastructure development has accelerated in recent years.
Major projects include the Kolkata Metro expansion, the Eastern Metropolitan
Bypass, the Kolkata Elevated Expressway, and the development of the
Kolkata Municipal Corporation area. The state has also invested in rural
road connectivity under the Pradhan Mantri Gram Sadak Yojana.

Bengali New Year, known as Poila Baisakh, is celebrated on the first day
of the Bengali calendar, usually falling on April 14 or 15. It is a major
cultural event marked by family gatherings, traditional food, new clothes,
and cultural programs. Rabindra Jayanti, the birth anniversary of Rabindranath
Tagore on May 8, is celebrated with music, dance, and literary events across
the state.

The state's relationship with Bangladesh is deep and complex. The 1971
Bangladesh Liberation War led to a massive influx of refugees into West Bengal.
The partition of Bengal in 1947 divided the Bengali-speaking population between
India and Pakistan (later Bangladesh). Cultural and family ties across the
border remain strong despite the political boundary.

Artisan communities in West Bengal have preserved traditional crafts for
generations. The Malakar community makes garlands and floral decorations.
The Kansari community works with bell metal. The Muchi community makes
leather goods. These traditional occupations are threatened by modernization,
and efforts are underway to preserve and promote them through government
schemes and fair trade organizations.

The Kolkata Book Fair, formally known as the International Kolkata Book Fair,
is the world's largest non-trade book fair and the most visited book fair
in the world. It is held annually in January-February and attracts millions
of visitors. The fair features publishers from across India and around the
world, and is a major cultural event in the Bengali calendar.

The architectural heritage of Kolkata includes numerous colonial-era buildings.
The Writer's Building, the seat of the state government, was built in 1777.
St. Paul's Cathedral, built in 1847, is an example of Gothic Revival
architecture. The Marble Palace, a 19th-century mansion, houses a vast
collection of art and antiques. These buildings reflect the city's rich
colonial history and architectural diversity.

Sports infrastructure in West Bengal includes several world-class facilities.
The Salt Lake Stadium, also known as Vivekananda Yuba Bharati Krirangan,
has a capacity of over 85,000 and has hosted the SAFF Championship and
other international football events. The Eden Gardens, with a capacity of
over 66,000, is one of the most iconic cricket venues in the world. The
Netaji Indoor Stadium hosts various indoor sports events.

The state government has implemented numerous welfare schemes for different
sections of society. The Kanyashree Prakalpa provides scholarships to girls
from economically disadvantaged backgrounds to promote their education. The
Sabuj Sathi scheme provides bicycles to students to improve school attendance.
The Ladli S scheme provides financial assistance to families with girls.

West Bengal's contribution to Indian art is immense. The Bengal School of
Art, founded by Abanindranath Tagore, was a movement that sought to develop
a distinctly Indian style of painting in response to the Western academic
style. Nandalal Bose, a student of Abanindranath, was one of the pioneers
of modern Indian art. Jamini Roy, another Bengal artist, is known for his
distinctive style inspired by folk art traditions.

The Sundarbans is not only a biodiversity hotspot but also a lifeline for
millions of people who depend on its resources for fishing, honey collection,
and wood. The forest provides natural protection against cyclones and tidal
waves. However, rising sea levels due to climate change pose a serious threat
to the Sundarbans ecosystem and the communities that depend on it.

The state's cultural calendar is packed with festivals and events throughout
the year. Besides Durga Puja, other major celebrations include Saraswati
Puja in February, Holi in March, Charak Puja in June, and Jagaddhatri Puja
in November. Each festival has its own unique traditions and cultural
significance, reflecting the syncretic nature of Bengali culture.

Tourism in West Bengal offers diverse experiences. The hill stations of
Darjeeling, Kalimpong, and Kurseong attract nature lovers. The beaches of
Digha and Mandarbani are popular with families. The historical sites of
Bishnupur and Murshidabad offer glimpses of the state's royal past. The
Sundarbans attract wildlife enthusiasts and adventure seekers from around
the world.

The state's industrial sector includes jute mills, tea processing units,
engineering factories, and IT companies. The jute industry, centered around
the Hooghly River, was once the backbone of the Bengal economy. Although
the industry has declined, efforts are being made to revive it through
innovation and new product development. The state is also exploring
opportunities in renewable energy, particularly solar and wind power.

Bengali literature continues to thrive in the modern era. Contemporary
writers like Mahasweta Devi, who wrote extensively about tribal communities,
and Sunil Gangopadhyay, known for his historical novels, have made significant
contributions. The annual Duttaput Literary Award and other prizes recognize
excellence in Bengali literature. Book clubs and literary circles remain an
integral part of Bengali urban life.

The state's cuisine varies by region. In the north, the food is influenced
by Tibetan and Nepali cuisine, with momos and thukpa being popular. In the
south, the Sundarbans area is known for its crab and prawn preparations.
In the east, the Malda district is famous for its mangoes. The Murshidabad
district is known for its silk and the famous Murshidabadi biryani.

West Bengal's educational initiatives include the Mid-Day Meal Scheme, which
provides free lunch to school children. The state has also launched digital
classrooms and smart schools to modernize education. The Sabar Shiksha
project aims to provide education to children from tribal and marginalized
communities in remote areas.

The state has a rich tradition of philosophical thought. The Navavidhan
movement, led by Swami Vivekananda (born Narendranath Datta in Kolkata),
promoted the synthesis of Eastern and Western philosophy. Sri Ramakrishna,
Vivekananda's guru, practiced and preached the harmony of all religions.
The Belur Math, the headquarters of the Ramakrishna Mission near Kolkata,
is a symbol of religious unity and service.

The Hooghly River, a distributary of the Ganges, is the lifeline of Kolkata
and its surrounding areas. The river supports fishing, transportation, and
religious activities. The ghats along the river, like the Nimtala Ghat and
the Prinsep Ghat, are important cultural and religious sites. The river
also plays a central role in the Durga Puja immersion ceremony, when idols
of the goddess are immersed in its waters.

The state's natural resources include coal, iron ore, and limestone in the
western districts. The Jharia coalfield in West Bengal is one of the largest
coal mining areas in India. The state also has deposits of manganese, copper,
and other minerals. These resources support the industrial development of the
region.

Climate change poses significant challenges to West Bengal. Rising sea levels
threaten the Sundarbans and coastal communities. Changing rainfall patterns
affect agriculture. The state government has taken measures to address these
challenges, including the construction of embankments, promotion of climate-resilient
agriculture, and participation in national and international climate change
initiatives.

The state's cultural institutions include the Academy of Fine Arts, the
Bangla Academy, and the Paschimbanga Bangla Akademi. These institutions
promote art, literature, and cultural exchange. The Indian Museum, the
Asiatic Society, and the National Library are important research and
cultural institutions located in Kolkata.

West Bengal's contributions to Indian cuisine include the popularization
of street food in Kolkata. Kusum rolls, puchka (known as panipuri in
other parts of India), jhal muri (spiced puffed rice), and egg chop are
beloved street foods. The city's food culture is vibrant and diverse,
with restaurants ranging from traditional Bengali eateries to modern
fusion cuisine establishments.

The state government has implemented various schemes for rural development.
The MGNREGA scheme provides employment guarantee to rural workers. The
Swabhiman scheme aims to empower women through self-help groups. The
Gorkhaland Territorial Administration provides autonomous governance to
the hill regions of Darjeeling.

The Kolkata Municipal Corporation, established in 1876, is one of the oldest
municipal corporations in India. It governs the city of Kolkata and is
responsible for providing civic services to its residents. The corporation
has launched several smart city initiatives, including digital governance
and improved public transportation.

The artistic heritage of Bengal includes the Kalighat school of painting,
which emerged in the 19th century near the Kalighat temple. These paintings
depicted scenes from daily life and mythology with bold lines and vivid
colors. The Patachitra tradition of scroll painting is another important
art form, particularly in the Medinipur district.

West Bengal's educational achievements include producing several Nobel
laureates and world-renowned scholars. Amartya Sen (Economics, 1998),
Rabindranath Tagore (Literature, 1913), and Abhijit Banerjee (Economics,
2019) are among the most famous. The state's emphasis on education and
intellectual discourse has created a rich intellectual tradition that
continues to thrive.

The state's transportation network includes the Kolkata Metro, which is
the first metro rail system in India. The metro currently operates on
two lines, with several extensions under construction. The Kolkata Tram
is one of the few surviving tram systems in India, operating along several
routes in the city. The suburban railway network connects Kolkata with
its satellite towns and is used by millions of commuters daily.

The Sundarbans Biosphere Reserve covers an area of 9,630 square kilometers
and is one of the largest biosphere reserves in India. It was designated
a UNESCO Biosphere Reserve in 2001. The reserve is home to 260 species
of birds, 49 species of mammals, and numerous species of fish, amphibians,
and reptiles. Conservation efforts are crucial for protecting this unique
ecosystem from the threats of climate change and human encroachment.

Bengal's textile tradition includes the famous Jamdani weaving technique,
which produces intricate designs on muslin fabric. The Tant saree, a
traditional cotton saree, is an essential part of a Bengali woman's wardrobe.
The Baluchari saree from Burdwan features elaborate woven narratives from
mythology. These textile traditions represent centuries of artistic
excellence and cultural identity.

The state's contribution to Indian theater is significant. The Bengali
theater tradition dates back to the 18th century and includes both
professional and amateur productions. The Star Theatre and Minerva
Theatre in Kolkata were important venues for theatrical performances
during the early 20th century. Today, groups like Nandikar and
Rangakarmee continue the tradition of meaningful and innovative theater.

The natural beauty of West Bengal ranges from the snow-capped peaks of
the Himalayas to the sun-kneaded beaches of the Bay of Bengal. The
Dooars region, between the Teesta and Sankosh rivers, is known for its
tea gardens, forests, and wildlife. The Corbett National Park-like
Jaldapara and Buxa tiger reserves protect the rich biodiversity of the
northern districts.

The culinary diversity of Bengal extends to its sweet traditions. Bengali
sweets, known as mishti, are famous worldwide. Rosogolla, the most famous
Bengali sweet, is made from cottage cheese balls soaked in sugar syrup.
Sandesh is another popular sweet made from fresh cottage cheese with
various flavorings. The sweet shops of Kolkata, particularly in the
Bagbazar and Kumartuli areas, are legendary for their offerings.

The intellectual tradition of Bengal is reflected in its numerous libraries
and reading rooms. The National Library of India, located in Kolkata, is
the largest library in the country. The State Central Library and numerous
public libraries across the state promote reading and lifelong learning.
Bengalis are traditionally known for their love of books and intellectual
pursuits, and the tradition of Adda, informal intellectual discussions,
is an integral part of Bengali culture.

The state's environmental conservation efforts include the protection of
the Sundarbans, the conservation of the one-horned rhinoceros in Jaldapara,
and the preservation of the red panda in the eastern Himalayas. The
Wildlife Institute of India has several projects in West Bengal focused
on species conservation and habitat management.

The IT revolution in Kolkata has transformed the city's economy. The
state government has established several IT parks and software technology
parks to attract investment. The Bengal Silicon Valley Hub at Newtown
is expected to become a major IT destination. The state's IT exports
have grown significantly in recent years, contributing to the overall
economic development of the region.

Bengal's musical heritage extends beyond Rabindra Sangeet and Baul music.
The state has produced renowned classical musicians, including Ravi Shankar
(sitar), Vilayat Khan (sitar), and Ali Akbar Khan (sarod). The ITC Sangeet
Research Academy in Kolkata is a premier institution for the preservation
and promotion of Indian classical music. The Dover Lane Music Conference
is one of the most prestigious classical music festivals in India.

The state's governance has evolved significantly since independence. From
the early congress governments to the left front rule and the current
Trinamool Congress government, West Bengal has witnessed diverse political
experiments. The land reforms implemented during the left front era were
among the most significant in Indian history, redistributing land from
zamindars to tillers.

The state's judicial system includes the Calcutta High Court, one of the
oldest high courts in India, established in 1862. The court has played a
significant role in shaping Indian jurisprudence. The state also has a
network of district courts and subordinate courts that administer justice
at the grassroots level.

West Bengal's contribution to Indian cinema extends beyond Satyajit Ray.
The state has produced acclaimed actors like Uttam Kumar, Suchitra Sen,
and more recently, Prosenjit Chatterjee and Jeet. The film industry has
also given rise to talented technicians, music directors, and scriptwriters
who have contributed to both Bengali and Hindi cinema.

The state's tourism infrastructure includes a range of accommodation options,
from luxury hotels to budget guest houses. The West Bengal Tourism Development
Corporation operates several hotels and resorts across the state. Homestays
have become increasingly popular, particularly in the Dooars, Darjeeling,
and Sundarbans regions, offering tourists an authentic cultural experience.

The traditional medicine systems of Bengal include Ayurveda, Unani, and
Siddha. The state has several institutions that teach and practice these
traditional systems of medicine. The National Institute of Naturopathy in
Pune has connections to Bengal's naturopathy tradition, which was promoted
by Mahatma Gandhi.

The state's agricultural diversity includes the cultivation of rice, jute,
tea, potatoes, and various fruits. The Alphonso mangoes of Malda, the
lychees of Murshidabad, and the pineapples of Jalpaiguri are famous for
their quality. The state is also a major producer of vegetables, particularly
in the northern districts.

The cultural exchange between West Bengal and Bangladesh continues despite
the political boundary. Literature, music, and cinema flow freely between
the two Bengals. The shared cultural heritage is celebrated through
festivals, literary events, and cultural programs on both sides of the border.

The state's infrastructure development has focused on improving connectivity.
The Bogibeel Bridge in Assam, the Kolkata Metro extensions, and the
development of new highways have improved transportation links. The state
is also exploring the possibility of high-speed rail connectivity with
other major Indian cities.

The people of Bengal are known for their warmth, hospitality, and love of
culture. The tradition of Adda, informal group discussions over tea, is a
cherished social institution. Bengalis are known for their passion for food,
literature, music, and football. The phrase Para is an integral part of
Bengali identity, referring to the neighborhood community that forms the
social fabric of Bengali life.

The state's contribution to Indian philosophy includes the Advaita Vedanta
tradition of the Ramakrishna Mission, the Vaishnavism of Chaitanya
Mahaprabhu, and the Baul philosophy of inner realization. These diverse
philosophical traditions reflect the syncretic and inclusive nature of
Bengali culture.

The development trajectory of West Bengal in the 21st century is marked by
ambitious plans for economic growth, infrastructure modernization, and
social development. The state aims to balance industrial growth with
environmental sustainability and social equity. The vision for West Bengal
is to emerge as a leading state in eastern India while preserving its rich
cultural heritage and natural beauty.
"""

def main():
    out_path = os.path.join(os.path.dirname(__file__), "westbengal.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(DATASET.strip())
    size = os.path.getsize(out_path)
    chars = len(DATASET.strip())
    words = len(DATASET.split())
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"Dataset created: westbengal.txt")
    print(f"  Size: {size:,} bytes ({size/1024:.1f} KB)")
    print(f"  Characters: {chars:,}")
    print(f"  Words: {words:,}")

if __name__ == "__main__":
    main()
