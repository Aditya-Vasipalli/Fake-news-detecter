from shap_api import bert_predict_proba, analyze_text_with_sliding_window

# Test with a long text (simulating a long article)
long_text = """
Breaking News: Scientists at MIT have made a groundbreaking discovery that could revolutionize renewable energy. 
The research team, led by Dr. Sarah Johnson, has developed a new type of solar panel that achieves 95% efficiency, 
far exceeding current commercial panels which typically reach only 20-22% efficiency.

The breakthrough comes from a novel approach using quantum dots embedded in perovskite materials. These quantum dots 
can capture photons across a much broader spectrum of light, including infrared radiation that traditional solar 
panels cannot utilize. "This is the most significant advancement in photovoltaic technology since the invention of 
the silicon solar cell," said Dr. Johnson during a press conference at MIT yesterday.

The new panels, dubbed "Quantum Enhanced Photovoltaic Cells" (QEPCs), have undergone rigorous testing over the past 
two years. Initial laboratory results showed promise, but the team needed to verify that the technology could work 
in real-world conditions. Field tests conducted in various climates from the sunny deserts of Arizona to the cloudy 
regions of Northern Europe consistently showed efficiency rates above 90%.

One of the most remarkable aspects of this technology is its performance in low-light conditions. Traditional solar 
panels become significantly less effective during cloudy weather or in the early morning and late evening hours. 
The QEPCs, however, maintain over 70% of their peak efficiency even in these challenging conditions, thanks to their 
ability to harness infrared radiation and scattered light.

The environmental implications of this breakthrough are staggering. Current estimates suggest that if just 30% of 
global rooftops were covered with these high-efficiency panels, it could provide enough electricity to power the 
entire world's energy needs. This would dramatically reduce our reliance on fossil fuels and could be a game-changer 
in the fight against climate change.

Dr. Johnson's team is already working with several major manufacturers to begin commercial production. The first 
commercial QEPCs are expected to hit the market within 18 months, with initial production costs estimated to be only 
15% higher than current premium solar panels. Industry analysts predict that as production scales up, costs could 
actually become lower than traditional panels due to the higher energy output per panel.

The breakthrough has attracted significant investment from both government agencies and private investors. The 
Department of Energy has announced a $500 million grant to accelerate research and development, while tech giant 
Tesla has reportedly signed an exclusive licensing deal worth $2 billion to incorporate the technology into their 
solar roof products.

However, not everyone in the industry is convinced. Some skeptics argue that the laboratory results may not translate 
to real-world performance over the long term. Dr. Michael Chen, a solar energy expert at Stanford University, cautions 
that "while these results are impressive, we need to see how these panels perform after 20-25 years of operation, 
which is the typical lifespan we expect from solar installations."

Despite the skepticism, the scientific community has largely embraced the findings. The research has been peer-reviewed 
and published in the prestigious journal Nature Energy, lending credibility to the claims. Several independent 
laboratories have already begun attempts to replicate the results, with early reports suggesting successful reproduction 
of the key findings.

The technology could also have applications beyond traditional solar panels. Researchers are exploring its potential 
use in solar-powered vehicles, portable devices, and even space applications where high efficiency is crucial due to 
weight and space constraints. NASA has already expressed interest in testing the technology for future Mars missions.

As the world continues to grapple with climate change and the urgent need to transition to renewable energy sources, 
this breakthrough represents a beacon of hope. If the technology lives up to its promise, it could accelerate the 
global transition to clean energy by decades, potentially making fossil fuels obsolete much sooner than previously 
anticipated.
""" * 2  # Double the text to make it even longer

print("Testing sliding window with long article...")
print(f"Text length: {len(long_text)} characters ({len(long_text.split())} words)")

# Test the sliding window approach
result = bert_predict_proba([long_text])
print(f"\nFinal result: Fake={result[0][0]:.3f}, Real={result[0][1]:.3f}")

# Determine prediction
prediction = "FAKE NEWS" if result[0][0] > result[0][1] else "REAL NEWS"
confidence = max(result[0][0], result[0][1])
print(f"Prediction: {prediction} (Confidence: {confidence:.1%})")
