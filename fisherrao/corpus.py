"""
A small but deliberately diverse evaluation corpus.

WHY THIS EXISTS, AND ITS LIMITS
-------------------------------
Everything measured up to this point came from ONE 20-token sentence at ONE
token position.  That is a pipeline check, not evidence.  This module raises the
sample to several hundred (layer, position) pairs across domains, which is the
minimum for a layer-wise profile to mean anything.

It was written because HuggingFace was unreachable from this machine.  That
block is now FIXED (fisherrao/net.py -- the TLS interception was Avast, and
Python's verification now goes through the OS), so `sentences(source="wikitext")`
below pulls WikiText-103 validation and every runner switches corpus without
being edited.  Set FISHERRAO_CORPUS=wikitext and re-run.

The hand-written sentences remain the default, and their limits are real:

  * they are short (10-30 tokens), so long-range context effects are absent;
  * they are clean edited prose, unlike web text;
  * n is hundreds, not tens of thousands;
  * the author chose them, so they are not a random sample of anything.

  !!  BUT THE HEADLINE RESULTS HAVE NOW BEEN CHECKED AGAINST WIKITEXT  !!

07-stage4-log.md S1.3, n=108 per arm, same protocol both sides:

    E2   "no instrument tracks any other" HOLDS, and gets cleaner -- largest
         cross-instrument |rho| falls from 0.242 to 0.108, positive control
         K vs R stays at +0.691.
    RQ3a intrinsic still beats every proxy, by 1.9x on WikiText.
    NEW  intrinsic K's correlation with entropy moves by 0.007 between the two
         corpora; every proxy moves by 0.08-0.30, and King's angle changes SIGN.
         The intrinsic instrument is corpus-invariant and the proxies are not.

So results from this corpus are no longer "preliminary pending a real corpus"
for E2 and RQ3a specifically.  They still are for anything not on that list --
notably RQ3b, whose polysemy pairs are hand-written by construction and have no
WikiText equivalent (a sense-annotated corpus such as SemCor or WiC is the
route, and is still outstanding).
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# General prose, spread across registers and domains.
# ---------------------------------------------------------------------------
GENERAL = [
    "The committee met on Tuesday to review the budget for the coming year.",
    "Water expands when it freezes, which is why pipes burst in winter.",
    "She had never seen the ocean before, and stood silent at the edge.",
    "The compiler reported an error on line forty-two of the source file.",
    "Historians disagree about the causes of the collapse of the empire.",
    "Add the flour slowly, stirring constantly to avoid forming lumps.",
    "The patient's fever subsided after the second dose of antibiotics.",
    "Interest rates rose sharply, and the housing market cooled within months.",
    "He argued that the premise was false, so the conclusion did not follow.",
    "The satellite transmitted images of the storm as it crossed the coast.",
    "Most of the sand on that beach was carried there by the river.",
    "The violin was built in Cremona sometime in the early eighteenth century.",
    "Bacteria in the soil convert nitrogen into a form plants can absorb.",
    "After the verdict, the courtroom emptied slowly and without much noise.",
    "The algorithm runs in linear time provided the input is already sorted.",
    "Snow had fallen overnight, and the road out of the valley was closed.",
    "Her second novel sold poorly, though critics preferred it to the first.",
    "The engine failed twice during testing before the design was changed.",
    "Light from that galaxy has been travelling toward us for eight billion years.",
    "They planted the orchard on the south slope, where the frost came latest.",
]

# ---------------------------------------------------------------------------
# Polysemy minimal pairs for RQ3b.
#
#   !!  THE DISAMBIGUATING CONTEXT MUST PRECEDE THE AMBIGUOUS WORD  !!
#
# A first version of this list put the context AFTER the target ("The bank was
# steep..." vs "The bank refused..."), which made the experiment vacuous.  With
# CAUSAL attention the hidden state at `bank` attends only to `The bank` --
# identical in both sentences -- so the two states are bit-for-bit equal and every
# separation measure returned exactly 0.0 at every layer.  Later context cannot
# reach back to an earlier token.
#
# So each sentence below establishes the sense first and places the ambiguous word
# LAST, where its representation has seen the disambiguating context.
# run_polysemy.py asserts non-zero final separation to stop this recurring.
#
# SIZE.  RQ3b is the strongest result in the project (a 19-layer disagreement
# between metrics) and it rested on the first TEN pairs below.  The half-
# separation layer was stable across them but the steepest-rise layer had sd
# 10.2, which at n=10 is an interval too wide to publish.  The set is now 60
# pairs.  Stage 5 task 5.1 asks for >= 200 across >= 4 domains; 60 is the
# honest limit of what one author can write without the quality dropping, and
# it cuts the standard error by ~2.4x.  Treat 200 as still outstanding, to be
# reached from a sense-annotated corpus (SemCor / WiC) now that downloads work.
#
# EVERY PAIR MUST SATISFY, and these are checked by corpus.validate():
#   * both sentences end with the target word
#   * the target word appears exactly once, at the end
#   * the preceding context alone fixes the sense
#
# POLYSEMY_SENSES labels the two senses by domain, in the same order as the two
# sentences, so results can be grouped by contrast type rather than pooled.
# HETERONYMS lists the pairs whose two senses differ in PRONUNCIATION as well as
# meaning (bass, bow, lead).  Those are arguably not one word at all, and a
# subword tokeniser cannot see the difference -- keep them separable so the
# analysis can report with and without them.
# ---------------------------------------------------------------------------
POLYSEMY: list[tuple[str, str, str]] = [
    ("bank", "Kneeling in the wet grass beside the flooded river, she studied the bank",
             "After reviewing the mortgage paperwork all morning, she finally called the bank"),
    ("bat", "Deep inside the limestone cave the zoologist photographed a hanging bat",
            "Walking out to the crease in the failing light, he lifted his bat"),
    ("spring", "Cold water trickled from the mossy rocks where we found the spring",
               "The clockmaker opened the case and replaced the corroded metal spring"),
    ("bark", "He ran his palm over the rough trunk of the oak, feeling the bark",
             "The neighbours complained for weeks about that restless dog and its bark"),
    ("pupil", "The ophthalmologist dimmed the lamp and shone a light into the pupil",
              "At the assembly the headmaster praised his most diligent pupil"),
    ("crane", "At the construction site they hoisted the steel girder with a crane",
              "Wading slowly through the reeded marsh, the birdwatcher spotted a crane"),
    ("mint", "She crushed a handful of fragrant leaves from the herb bed, fresh mint",
             "The treasury ordered the commemorative coins struck at the royal mint"),
    ("plant", "In the humid greenhouse he repotted a wilting tomato plant",
              "Four hundred workers lost their jobs when the company closed the plant"),
    ("seal", "Out on the drifting ice floe the marine biologist tagged a seal",
             "He broke the brittle red wax on the envelope and examined the seal"),
    ("pitch", "The groundsman spent the morning rolling the sodden cricket pitch",
              "The soprano strained upward until she could not hold the pitch"),
    ("club", "The bouncer unhooked the velvet rope and waved us into the night club",
             "He wiped the grass from the head of his nine iron golf club"),
    ("match", "She struck the rough side of the box and lit a match",
              "Both sides walked back out for the second half of the match"),
    ("organ", "The surgeon explained that the liver is the body's largest internal organ",
              "The cathedral filled with sound as the musician settled at the organ"),
    ("note", "The cellist drew the bow slowly and sustained one long note",
             "She left the rent money on the counter with a handwritten note"),
    ("scale", "The fishmonger's blade slid beneath a silvery translucent scale",
              "Before weighing the parcel the clerk carefully zeroed the scale"),
    ("bolt", "Thunder arrived some seconds after the blinding lightning bolt",
             "The mechanic worked penetrating oil into the last rusted bolt"),
    ("current", "Swimmers were warned that morning about the strong offshore current",
                "The technician probed the circuit and then measured the current"),
    ("trunk", "The elephant reached up into the high branches with its trunk",
              "He rearranged the suitcases and slammed shut the car trunk"),
    ("ruler", "The dynasty remembered him as a just and long reigning ruler",
              "She drew the line carefully along the bevelled edge of the ruler"),
    ("bridge", "Traffic backed up for miles on the approach to the suspension bridge",
               "The dentist filled the gap in his upper jaw with a bridge"),
    ("cell", "Under the microscope the biologist watched a single dividing cell",
             "The guard turned the key and locked the prisoner in his cell"),
    ("court", "The judge entered and everyone rose in the crowded court",
              "The groundstaff swept fallen leaves from the clay tennis court"),
    ("fan", "In the stifling August heat she reached up and switched on the fan",
            "He had followed the club since childhood, a lifelong devoted fan"),
    ("file", "The carpenter smoothed the burr on the metal edge with a file",
             "The clerk retrieved the missing paperwork from the cabinet file"),
    ("jam", "She spread butter on the warm scone and then strawberry jam",
            "The motorway sat at a standstill in a five mile jam"),
    ("key", "He turned the lock twice and withdrew the heavy brass key",
            "The accompanist transposed the aria into a brighter key"),
    ("mole", "The dermatologist photographed a small dark irregular mole",
             "Fresh mounds of earth across the lawn betrayed a mole"),
    ("nail", "The carpenter set the board and drove in one more galvanised nail",
             "The manicurist shaped the edge of her broken thumb nail"),
    ("palm", "The fortune teller traced the deep lines across his open palm",
             "Green coconuts hung in the crown of the tallest palm"),
    ("pen", "She uncapped the heavy black fountain pen",
            "The farmer whistled the last of the ewes into the pen"),
    ("pole", "The expedition dragged its sledges toward the magnetic south pole",
             "The fisherman leaned his nets against a weathered wooden pole"),
    ("port", "The loaded freighter waited three days for a berth in the port",
             "After dinner the host poured each guest a small glass of port"),
    ("ring", "The jeweller resized the engraved gold wedding ring",
             "The boxers touched gloves in the middle of the ring"),
    ("rock", "The climber tested a loose handhold on the weathered granite rock",
             "The festival opened with three hours of straightforward guitar rock"),
    ("root", "The gardener's spade caught on a thick woody root",
             "The student solved the quadratic and reported the positive root"),
    ("sole", "The cobbler stripped away the cracked and worn leather sole",
             "The chef filleted a fresh line caught Dover sole"),
    ("star", "Through the telescope he located a faint and very distant star",
             "Photographers crowded the barrier around the arriving film star"),
    ("tank", "The aquarist checked the temperature and salinity of the tank",
             "The infantry advanced in the shelter of a single armoured tank"),
    ("tie", "He straightened his collar and adjusted his narrow silk tie",
            "Neither side scored again and the match finished in a tie"),
    ("vault", "The manager spun the dial and swung open the steel vault",
              "The gymnast sprinted the length of the runway toward the vault"),
    ("wave", "Surfers paddled out beyond the break to meet the incoming wave",
             "The physicist described light as both particle and wave"),
    ("yard", "The children spent the afternoon playing in the fenced back yard",
             "The tailor unrolled the bolt and measured out one yard"),
    ("chest", "The doctor warmed the stethoscope and placed it on his chest",
              "In the attic they found a locked and banded wooden chest"),
    ("coach", "The players gathered at half time around their long serving coach",
              "The tour party filed off the airport shuttle coach"),
    ("deck", "The crew spent the morning scrubbing salt from the wooden deck",
             "The dealer shuffled twice and offered the cut of the deck"),
    ("draft", "The editor returned the manuscript covered in notes, a rough draft",
              "A cold air came in under the ill fitting door, a persistent draft"),
    ("drill", "The builder changed the masonry bit on his cordless drill",
              "The school assembled on the field for its termly fire drill"),
    ("grain", "The miller ran his fingers through the freshly threshed grain",
              "The joiner planed the board along the direction of the grain"),
    ("pitcher", "She filled the glazed earthenware pitcher",
                "The bullpen began warming up a left handed pitcher"),
    ("post", "The bills and a birthday card arrived in the morning post",
             "They tamped the concrete around the last fence post"),
    ("pound", "The stray was collected by the warden and taken to the pound",
              "The recipe called for butter and flour, half a pound"),
    ("punch", "The challenger dropped his guard and took a heavy right punch",
              "At the reception they ladled out a sweet fruit punch"),
    ("shell", "She stooped on the tideline and picked up a spiral pink shell",
              "The battery loaded and fired one more high explosive shell"),
    ("tablet", "The pharmacist counted the dose out and sealed one final tablet",
               "Archaeologists slowly deciphered the cuneiform on the clay tablet"),
    ("temple", "The pilgrims left their shoes on the steps of the temple",
               "He closed his eyes and pressed against his throbbing temple"),
    ("wing", "The engineer inspected a hairline crack in the aircraft wing",
             "Refurbishment had closed the whole of the hospital's east wing"),
    ("bill", "The waiter cleared the plates and quietly brought over the bill",
             "The pelican scooped up the fish in its enormous bill"),
    ("case", "The barrister worked all night on her closing argument for the case",
             "He loosened the bow, snapped shut the lid of the violin case"),
    ("board", "The carpenter sawed cleanly through the knotted pine board",
              "Shareholders voted two new directors onto the board"),
    ("capital", "The delegation spent three days of talks in the nation's capital",
                "The founders needed another round to raise working capital"),
    ("figure", "The accountant went back through the ledger and checked the figure",
               "Through the fog on the moor he made out a distant human figure"),
    ("bass", "The angler netted and weighed a five pound striped bass",
             "The rhythm section was anchored by a warm fretless bass"),
    ("bow", "The archer nocked an arrow and drew back his yew bow",
            "The applause swelled and the soloist rose to take a bow"),
    ("lead", "Victorian plumbers joined their pipes with molten lead",
             "The dog strained and coughed at the end of its lead"),
]

# ---------------------------------------------------------------------------
# SAME-SENSE minimal pairs -- the control the probe set was missing.
#
# WHY THIS EXISTS.  POLYSEMY above has only DIFFERENT-sense pairs, so it cannot
# answer the obvious challenge: do any two occurrences of a word in two
# different sentences separate the same way?  WiC supplies a same-sense arm and
# 07-stage4-log.md S3b.5 ran it -- with real preceding context, WiC's same-sense
# pairs separate 1.01-1.14x as much as its different-sense pairs, i.e. the
# instruments were tracking DIFFERENT CONTEXT rather than DIFFERENT SENSE.
#
# But WiC's control is too loose to settle it: its pairs are arbitrary sentences
# that differ in every way at once.  The tight control is same-sense pairs built
# to the SAME recipe as POLYSEMY -- same target word, same final position,
# matched syntactic frame, both sentences fixing the SAME sense.  Then the only
# thing that differs is the surface wording of the context.
#
# THE TEST THIS MAKES POSSIBLE:
#
#     different-sense separation  >>  same-sense separation   -> the instruments
#                                                                read sense
#     different-sense separation  ~=  same-sense separation   -> they read
#                                                                context, and
#                                                                S3b's semantic
#                                                                framing must go
#
# Each entry reuses sense A of the corresponding POLYSEMY word, so the two sets
# are directly comparable.  Same structural rules, checked by validate().
# ---------------------------------------------------------------------------
POLYSEMY_SAME: list[tuple[str, str, str]] = [
    ("bank", "Kneeling in the wet grass beside the flooded river, she studied the bank",
             "Crouching among the damp reeds along the swollen stream, he examined the bank"),
    ("bat", "Deep inside the limestone cave the zoologist photographed a hanging bat",
            "High in the hollow oak the naturalist filmed a roosting bat"),
    ("spring", "Cold water trickled from the mossy rocks where we found the spring",
               "Clear water bubbled between the ferns where they located the spring"),
    ("bark", "He ran his palm over the rough trunk of the oak, feeling the bark",
             "She pressed her fingers against the ridged stem of the elm, feeling the bark"),
    ("pupil", "The ophthalmologist dimmed the lamp and shone a light into the pupil",
              "The optician lowered the blind and directed a beam into the pupil"),
    ("crane", "At the construction site they hoisted the steel girder with a crane",
              "On the dockside they lifted the loaded container with a crane"),
    ("mint", "She crushed a handful of fragrant leaves from the herb bed, fresh mint",
             "He tore a few aromatic sprigs from the kitchen garden, fresh mint"),
    ("plant", "In the humid greenhouse he repotted a wilting tomato plant",
              "Inside the warm conservatory she watered a drooping pepper plant"),
    ("seal", "Out on the drifting ice floe the marine biologist tagged a seal",
             "Along the rocky haul out the field researcher photographed a seal"),
    ("pitch", "The groundsman spent the morning rolling the sodden cricket pitch",
              "The curator spent the afternoon marking the muddy football pitch"),
    ("club", "The bouncer unhooked the velvet rope and waved us into the night club",
             "The doorman lifted the barrier and ushered them into the jazz club"),
    ("match", "She struck the rough side of the box and lit a match",
              "He scraped the flint strip and ignited a match"),
    ("organ", "The surgeon explained that the liver is the body's largest internal organ",
              "The anatomist noted that the kidney is a vital filtering organ"),
    ("note", "The cellist drew the bow slowly and sustained one long note",
             "The flautist held her breath steady and sustained one clear note"),
    ("scale", "The fishmonger's blade slid beneath a silvery translucent scale",
              "The angler's thumbnail lifted a small iridescent scale"),
    ("bolt", "Thunder arrived some seconds after the blinding lightning bolt",
             "The rumble followed moments behind the searing lightning bolt"),
    ("current", "Swimmers were warned that morning about the strong offshore current",
                "Bathers were cautioned at noon about the fierce outgoing current"),
    ("trunk", "The elephant reached up into the high branches with its trunk",
              "The bull pulled down a low hanging branch with its trunk"),
    ("ruler", "The dynasty remembered him as a just and long reigning ruler",
              "The chronicle described her as a wise and widely respected ruler"),
    ("bridge", "Traffic backed up for miles on the approach to the suspension bridge",
               "Queues stretched for hours at the entrance to the cantilever bridge"),
    ("cell", "Under the microscope the biologist watched a single dividing cell",
             "Through the eyepiece the technician observed a single stained cell"),
    ("court", "The judge entered and everyone rose in the crowded court",
              "The magistrate arrived and the gallery stood in the packed court"),
    ("fan", "In the stifling August heat she reached up and switched on the fan",
            "During the airless July afternoon he leaned over and started the fan"),
    ("file", "The carpenter smoothed the burr on the metal edge with a file",
             "The locksmith worked the rough notch on the brass blank with a file"),
    ("jam", "She spread butter on the warm scone and then strawberry jam",
            "He layered cream on the fresh scone and then raspberry jam"),
    ("key", "He turned the lock twice and withdrew the heavy brass key",
            "She released the deadbolt and pocketed the small iron key"),
    ("mole", "The dermatologist photographed a small dark irregular mole",
             "The nurse measured a raised brown asymmetric mole"),
    ("nail", "The carpenter set the board and drove in one more galvanised nail",
             "The joiner braced the plank and hammered in another steel nail"),
    ("palm", "The fortune teller traced the deep lines across his open palm",
             "The nurse pressed a folded cloth into her upturned palm"),
    ("pen", "She uncapped the heavy black fountain pen",
            "He unscrewed the slim silver fountain pen"),
    ("port", "The loaded freighter waited three days for a berth in the port",
             "The rusting tanker lay two weeks awaiting clearance in the port"),
    ("ring", "The jeweller resized the engraved gold wedding ring",
             "The goldsmith polished the inherited silver signet ring"),
]


# ---------------------------------------------------------------------------
# FRAME-MATCHED different-sense pairs -- closing the confound in POLYSEMY_SAME.
#
# 07-stage4-log.md S3b.6a found that POLYSEMY_SAME's B sentences share 2.00x
# more vocabulary with A than POLYSEMY's B sentences do (Jaccard 0.235 vs 0.118,
# closer in 30/32 words, z = +4.95).  That is the "matched syntactic frame"
# requirement doing it: reusing A's frame reuses its function words.  So the two
# arms differed in sense AND in lexical overlap, and a 2.00x overlap difference
# sitting next to a 2.19x separation difference is not a conclusion.
#
# This list fixes it from the other side: the SAME sentence A, paired with a
# different-sense B written to A's frame as closely as the same-sense B is.
# Overlap is then equalised by construction and validate() checks it.
#
#     POLYSEMY[w]        A  vs  different-sense B, free frame   (published set)
#     POLYSEMY_SAME[w]   A  vs  same-sense B, A's frame
#     POLYSEMY_DIFF[w]   A  vs  different-sense B, A's frame    <- this list
#
# The clean comparison is POLYSEMY_DIFF against POLYSEMY_SAME: same A, same
# frame, same overlap, differing only in sense.
# ---------------------------------------------------------------------------
POLYSEMY_DIFF: list[tuple[str, str, str]] = [
    ("bank", "Kneeling in the wet grass beside the flooded river, she studied the bank",
             "After a long morning of mortgage paperwork she finally telephoned the bank"),
    ("bat", "Deep inside the limestone cave the zoologist photographed a hanging bat",
            "Out on the floodlit square the opening batsman lifted a heavy bat"),
    ("spring", "Cold water trickled from the mossy rocks where we found the spring",
               "Thin oil seeped from the cracked casing around the corroded spring"),
    ("bark", "He ran his palm over the rough trunk of the oak, feeling the bark",
             "He heard his neighbour's dog over the fence answer with a sharp bark"),
    ("pupil", "The ophthalmologist dimmed the lamp and shone a light into the pupil",
              "The headmaster dimmed the lamp and turned a question to the pupil"),
    ("crane", "At the construction site they hoisted the steel girder with a crane",
              "At the reeded lakeside they photographed the stalking grey crane"),
    ("mint", "She crushed a handful of fragrant leaves from the herb bed, fresh mint",
             "She collected a handful of gleaming coins from the treasury vault, struck at the mint"),
    ("plant", "In the humid greenhouse he repotted a wilting tomato plant",
              "After the third quarter of losses they closed the assembly plant"),
    ("seal", "Out on the drifting ice floe the marine biologist tagged a seal",
             "Down on the folded parchment the archivist examined a seal"),
    ("pitch", "The groundsman spent the morning rolling the sodden cricket pitch",
              "The soprano spent the morning holding a difficult high pitch"),
    ("club", "The bouncer unhooked the velvet rope and waved us into the night club",
             "The caddie unzipped the leather bag and handed him the iron club"),
    ("match", "She struck the rough side of the box and lit a match",
              "She reached the closing stage of the set and won the match"),
    ("organ", "The surgeon explained that the liver is the body's largest internal organ",
              "The organist explained that the cathedral holds the country's largest pipe organ"),
    ("note", "The cellist drew the bow slowly and sustained one long note",
             "The lodger folded the paper slowly and left one short note"),
    ("scale", "The fishmonger's blade slid beneath a silvery translucent scale",
              "The chemist's parcel settled onto a freshly zeroed scale"),
    ("bolt", "Thunder arrived some seconds after the blinding lightning bolt",
             "Rust flaked away moments after he loosened the seized bolt"),
    ("current", "Swimmers were warned that morning about the strong offshore current",
                "Technicians were cautioned about the sudden surge in the induced current"),
    ("trunk", "The elephant reached up into the high branches with its trunk",
              "The porter reached down into the packed car and closed the trunk"),
    ("ruler", "The dynasty remembered him as a just and long reigning ruler",
              "The draughtsman kept beside him a steel and finely graduated ruler"),
    ("bridge", "Traffic backed up for miles on the approach to the suspension bridge",
               "Discomfort lingered for weeks after the fitting of the dental bridge"),
    ("cell", "Under the microscope the biologist watched a single dividing cell",
             "Behind the iron door the warder watched a single occupied cell"),
    ("court", "The judge entered and everyone rose in the crowded court",
              "The champion entered and everyone cheered around the clay court"),
    ("fan", "In the stifling August heat she reached up and switched on the fan",
            "Among the roaring Saturday crowd stood one lifelong devoted fan"),
    ("file", "The carpenter smoothed the burr on the metal edge with a file",
             "The clerk smoothed the crease on the missing folder and opened a file"),
    ("jam", "She spread butter on the warm scone and then strawberry jam",
            "She spread maps on the warm bonnet and waited out the jam"),
    ("key", "He turned the lock twice and withdrew the heavy brass key",
            "She transposed the aria upward and settled on a brighter key"),
    ("mole", "The dermatologist photographed a small dark irregular mole",
             "The gardener uncovered a small fresh earthen mound left by a mole"),
    ("nail", "The carpenter set the board and drove in one more galvanised nail",
             "The manicurist held the hand and filed down one splitting nail"),
    ("palm", "The fortune teller traced the deep lines across his open palm",
             "The botanist traced the ringed scars up the tall coconut palm"),
    ("pen", "She uncapped the heavy black fountain pen",
            "She unlatched the muddy wooden sheep pen"),
    ("port", "The loaded freighter waited three days for a berth in the port",
             "After dinner the host decanted a small glass of ruby port"),
    ("ring", "The jeweller resized the engraved gold wedding ring",
             "The referee cleared the roped canvas boxing ring"),
]


# The two senses, in the same order as the two sentences above.
POLYSEMY_SENSES: dict[str, tuple[str, str]] = {
    "bank": ("nature", "institution"),   "bat": ("nature", "sport"),
    "spring": ("nature", "artifact"),    "bark": ("nature", "nature"),
    "pupil": ("body", "institution"),    "crane": ("artifact", "nature"),
    "mint": ("food", "institution"),     "plant": ("nature", "institution"),
    "seal": ("nature", "artifact"),      "pitch": ("sport", "arts"),
    "club": ("institution", "sport"),    "match": ("artifact", "sport"),
    "organ": ("body", "arts"),           "note": ("arts", "artifact"),
    "scale": ("nature", "science"),      "bolt": ("nature", "artifact"),
    "current": ("nature", "science"),    "trunk": ("nature", "artifact"),
    "ruler": ("institution", "artifact"), "bridge": ("artifact", "body"),
    "cell": ("nature", "institution"),   "court": ("institution", "sport"),
    "fan": ("artifact", "sport"),        "file": ("artifact", "institution"),
    "jam": ("food", "artifact"),         "key": ("artifact", "arts"),
    "mole": ("body", "nature"),          "nail": ("artifact", "body"),
    "palm": ("body", "nature"),          "pen": ("artifact", "nature"),
    "pole": ("science", "artifact"),     "port": ("artifact", "food"),
    "ring": ("artifact", "sport"),       "rock": ("nature", "arts"),
    "root": ("nature", "science"),       "sole": ("artifact", "food"),
    "star": ("science", "arts"),         "tank": ("nature", "artifact"),
    "tie": ("artifact", "sport"),        "vault": ("institution", "sport"),
    "wave": ("nature", "science"),       "yard": ("artifact", "science"),
    "chest": ("body", "artifact"),       "coach": ("sport", "artifact"),
    "deck": ("artifact", "arts"),        "draft": ("arts", "nature"),
    "drill": ("artifact", "institution"), "grain": ("food", "artifact"),
    "pitcher": ("artifact", "sport"),    "post": ("institution", "artifact"),
    "pound": ("institution", "science"), "punch": ("sport", "food"),
    "shell": ("nature", "artifact"),     "tablet": ("body", "arts"),
    "temple": ("institution", "body"),   "wing": ("artifact", "institution"),
    "bill": ("institution", "nature"),   "case": ("institution", "arts"),
    "board": ("artifact", "institution"), "capital": ("institution", "institution"),
    "figure": ("science", "arts"),       "bass": ("nature", "arts"),
    "bow": ("sport", "arts"),            "lead": ("artifact", "nature"),
}

# Same spelling, DIFFERENT pronunciation.  A subword tokeniser cannot
# distinguish these, so the model has no phonological cue -- report separately.
HETERONYMS = ("bass", "bow", "lead")


def validate() -> list[str]:
    """Structural checks on POLYSEMY.  Returns a list of problems; empty is good.

    Catches, mechanically, the failure mode that made the FIRST version of this
    experiment vacuous: the disambiguating context must come BEFORE the target,
    which under causal attention means the target must be the LAST word.  A pair
    that violates it returns exactly 0.0 separation at every layer and looks
    like a null result rather than a bug.
    """
    problems = []
    for label, pairs in (("POLYSEMY", POLYSEMY), ("POLYSEMY_SAME", POLYSEMY_SAME),
                         ("POLYSEMY_DIFF", POLYSEMY_DIFF)):
        seen = set()
        for word, a, b in pairs:
            if word in seen:
                problems.append(f"{label}/{word}: duplicate entry")
            seen.add(word)
            if label == "POLYSEMY" and word not in POLYSEMY_SENSES:
                problems.append(f"{label}/{word}: missing from POLYSEMY_SENSES")
            for tag, s in (("A", a), ("B", b)):
                toks = s.rstrip(".").lower().split()
                if toks[-1] != word:
                    problems.append(f"{label}/{word} [{tag}]: does not end with the "
                                    f"target (ends with {toks[-1]!r})")
                if toks.count(word) != 1:
                    problems.append(f"{label}/{word} [{tag}]: target appears "
                                    f"{toks.count(word)} times, must be exactly 1")
            if a == b:
                problems.append(f"{label}/{word}: the two sentences are identical")

    # The same-sense control is only a control if it uses the SAME words, and
    # its sentence A must be the SAME sentence as POLYSEMY's sentence A -- so
    # that the two arms share a reference point and differ only in what the
    # OTHER sentence does.
    poly = {w: (a, b) for w, a, b in POLYSEMY}
    for label, pairs in (("POLYSEMY_SAME", POLYSEMY_SAME),
                         ("POLYSEMY_DIFF", POLYSEMY_DIFF)):
        for word, a, _ in pairs:
            if word not in poly:
                problems.append(f"{label}/{word}: not a POLYSEMY word, so the "
                                f"arms are not comparable")
            elif a != poly[word][0]:
                problems.append(f"{label}/{word}: sentence A differs from "
                                f"POLYSEMY's sentence A; the arms lose their "
                                f"shared reference point")

    # THE CONFOUND CHECK.  POLYSEMY_SAME and POLYSEMY_DIFF exist to be compared,
    # so their lexical overlap with A must match -- that is the whole reason
    # POLYSEMY_DIFF was written (07-stage4-log.md S3b.6a).  Flag any word where
    # the two arms differ by more than 0.10 Jaccard, and flag the SET if the
    # medians drift apart.
    def _jac(x: str, y: str) -> float:
        A = {t.strip(".,").lower() for t in x.split()}
        B = {t.strip(".,").lower() for t in y.split()}
        return len(A & B) / len(A | B) if A | B else 0.0

    same_b = {w: b for w, _, b in POLYSEMY_SAME}
    ov_s, ov_d = [], []
    for word, a, b_diff in POLYSEMY_DIFF:
        if word not in same_b:
            continue
        js, jd = _jac(a, same_b[word]), _jac(a, b_diff)
        ov_s.append(js)
        ov_d.append(jd)
        if abs(js - jd) > 0.10:
            problems.append(f"overlap/{word}: same-sense {js:.3f} vs "
                            f"different-sense {jd:.3f} -- arms not frame-matched")
    if ov_s:
        ms = sorted(ov_s)[len(ov_s) // 2]
        md = sorted(ov_d)[len(ov_d) // 2]
        if abs(ms - md) > 0.04:
            problems.append(f"overlap/SET: median Jaccard same {ms:.3f} vs "
                            f"different {md:.3f} -- the arms are confounded with "
                            f"lexical overlap")
    return problems


def domains() -> dict[str, int]:
    """How many pairs touch each sense domain."""
    counts: dict[str, int] = {}
    for senses in POLYSEMY_SENSES.values():
        for s in senses:
            counts[s] = counts.get(s, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------
# WikiText-103 -- the corpus the hand-written one was standing in for.
#
# Every log in this project carries the caveat "40 hand-written sentences, not a
# random sample of anything".  That caveat existed because HuggingFace was
# unreachable (05-stage0-log.md S0), which is now fixed (fisherrao/net.py), so
# it can be retired by re-running rather than by argument.
#
# Only the VALIDATION split is fetched -- 657 KB, versus ~500 MB for the full
# corpus -- which is also the split Mabrok 2026 reports his curvature on, so
# comparisons against that paper are on the same text.
# ---------------------------------------------------------------------------
_WIKITEXT_CACHE: list[str] | None = None


def wikitext(n: int = 200, min_words: int = 8, max_words: int = 40,
             seed: int = 0) -> list[str]:
    """`n` sentences sampled deterministically from WikiText-103 validation.

    Sentences rather than paragraphs, and length-bounded, so that the per-item
    cost is comparable to the hand-written corpus and one very long article does
    not dominate the sample.  Headings (`= Title =`) and list fragments are
    dropped.
    """
    global _WIKITEXT_CACHE
    if _WIKITEXT_CACHE is None:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="Salesforce/wikitext",
            filename="wikitext-103-raw-v1/validation-00000-of-00001.parquet",
            repo_type="dataset")
        pool: list[str] = []
        for line in pq.read_table(path).column("text").to_pylist():
            s = line.strip()
            if not s or s.startswith("="):
                continue
            for part in s.split(". "):
                part = part.strip().rstrip(".")
                if min_words <= len(part.split()) <= max_words and part[:1].isupper():
                    pool.append(part + ".")
        _WIKITEXT_CACHE = pool

    import torch

    g = torch.Generator().manual_seed(seed)
    pick = torch.randperm(len(_WIKITEXT_CACHE), generator=g)[:n].tolist()
    return [_WIKITEXT_CACHE[i] for i in pick]


def sentences(include_polysemy: bool = True, source: str | None = None) -> list[str]:
    """All corpus sentences as a flat list.

    `source` (or the FISHERRAO_CORPUS environment variable) selects:

        handwritten   the 20 general + 128 polysemy sentences below (default)
        wikitext      WikiText-103 validation, `FISHERRAO_CORPUS_N` sentences
                      (default 200)

    Routing it through here means every runner switches corpus without being
    edited -- they all call corpus.sentences().  Set the variable, re-run, and
    compare; that is the whole point.
    """
    source = source or os.environ.get("FISHERRAO_CORPUS", "handwritten")
    if source == "wikitext":
        return wikitext(int(os.environ.get("FISHERRAO_CORPUS_N", "200")))
    if source != "handwritten":
        raise ValueError(f"unknown corpus source {source!r}; "
                         f"use 'handwritten' or 'wikitext'")
    out = list(GENERAL)
    if include_polysemy:
        for _, a, b in POLYSEMY:
            out.extend([a, b])
    return out


def summary() -> str:
    n_poly = 2 * len(POLYSEMY)
    return (f"corpus: {len(GENERAL)} general + {n_poly} polysemy "
            f"({len(POLYSEMY)} minimal pairs) = {len(GENERAL) + n_poly} sentences")
