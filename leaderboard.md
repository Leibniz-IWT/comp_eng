---
title: Project Leaderboard – RL-based Orbit Station-Keeping
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
---

# Project Leaderboard

**Last updated: 02 February 2026**

This leaderboard shows validated student submissions for the final course project (Unit 14: RL-based orbit station-keeping).

**Ranking criteria** (in order of priority):  
1. Days in orbit (higher = better sustained station-keeping)  
2. Fuel mass remaining (higher = better fuel efficiency)  
3. Score (average episodic reward)

Only submissions manually verified by the instructor are listed.  
Students should submit via the designated channel (Google Form / email) including:  
- Link to your notebook/Colab (publicly viewable)  
- Model files (shareable Nexcloud or Drive folder or GitHub release)  
- Self-reported metrics and a short description

| Rank | Model name                        | Score              | Days in orbit    | Fuel mass remaining (kg) | Notes                                                                                                                                                                                                                                                                |
|------|-----------------------------------|--------------------|------------------|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.   | I_Am_Really_Done_ok               | 238316.09242013458 | 180.0 days       | 329.72 kg                | Used PPO with custom reward shaping featuring an altitude deadzone (5km) for fuel efficiency and an alignment penalty for integrated control. The agent acts as a numerical decoupler for the hierarchical control scheme in Sub-Problem 3.                          |
| 2.   | Return_to_Sender_v16              | 235598.18129603358 | 180.0 days       | 325.57 kg                | –                                                                                                                                                                                                                                                                    |
| 3.   | Fawkes V18                        | 233299.97598548885 | 180.0 days       | 322.03 kg                | –                                                                                                                                                                                                                                                                    |
| 4.   | Houston_We_Have_A_Lot_Of_Problems | 232811.6211        | 180.0 days       | 321.27 kg                | Minimal change in PPO agent hyperparameters                                                                                                                                                                                                                          |
| 5.   | Iamconfused_V1                    | 229503.53107895696 | 180.0 days       | 316.08 kg                | -                                                                                                                                                                                                                                                                    |
| 6.   | AstraKeep_SC_1100k                | 229001.36          | 180.0 days       | 315.28 kg                | PPO-based high-level station-keeping controller with integrated reward shaping directly inside the environment definition. Instead of using the separate “Reward Shaping” section, I embedded a physics-informed energy-based reward into the PPO training workflow. |
| 7.   | CMπF_The_End                      | 226888.9319445231  | 180.0 days       | 311.91 kg                | -                                                                                                                                                                                                                                                                    |
| 8.   | Houston_We_Have_No_Problem        | 223453.60          | 180.0 days       | 306.35 kg                | –                                                                                                                                                                                                                                                                    |
| 9.   | Red_Falcon_V5                     | 213529.6644028123  | 180.0 days       | 289.71 kg                | –                                                                                                                                                                                                                                                                    |
| 10.  | AGENT ZORO                        | 186939.11961551246 | 180.0 days       | 239.46 kg                | –                                                                                                                                                                                                                                                                    |
| 11.  | minimum_effort_v2                 | 19265.44           | 180.0 days       | 0.0 kg                   | Minimum passing grade as defined by the rules                                                                                                                                                                                                                        |


This project presents a hybrid control architecture that merges classical engineering with Reinforcement Learning (PPO). The system uses an optimised low-level classical controller (featuring a feedforward decoupler and a 0.5° deadzone) to manage attitude stability and eliminate thruster jitter. Meanwhile, the trained PPO agent handles high-level altitude maintenance using a fuel-efficient "Pulse-Glide" strategy, successfully sustaining the orbit for the full 6-month requirement.
PS: I have done Subtask 3 and Subtask 5 in the same notebook "Fawkes_Subtask3". And I have already submitted the major task before (already on Leaderboard), i just wanted to submit all the files together in one email.

<script>
// Client-side sorting: click column headers (Score, Days in orbit, Fuel mass remaining)
document.addEventListener("DOMContentLoaded", function() {
  const table = document.querySelector("table");
  const headers = tablfrom IPython.display import Video

# Embed with base64 (works locally and on Pages, no path issues)
Video("animation.mp4", embed=True, width=600, html_attributes="controls loop")e.querySelectorAll("th");
  
  headers.forEach((header, index) => {
    if (index === 0 || index === 1 || index === 5) return; // skip Rank, Model name, Notes
    header.style.cursor = "pointer";
    header.onclick = () => {
      const rows = Array.from(table.querySelectorAll("tbody tr"));
      const asc = header.classList.toggle("asc");
      
      rows.sort((a, b) => {
        let valA = a.children[index].textContent.trim();
        let valB = b.children[index].textContent.trim();
        
        const numericA = parseFloat(valA) || 0;
        const numericB = parseFloat(valB) || 0;
        
        return asc ? numericA - numericB : numericB - numericA;
      });
      
      rows.forEach(row => table.querySelector("tbody").appendChild(row));
    };
  });
});
</script>

<style>
table th { position: relative; }
table th.asc::after { content: " ▼"; }
table th:not(.asc)::after { content: " ▲"; }
</style>

**Legend**  
- **Score**: Average episodic reward over standardised evaluation (higher is better)  
- **Days in orbit**: Average number of days the spacecraft remains within station-keeping bounds before violation (higher is better)  
- **Fuel mass remaining**: Remaining propellant mass (kg) at episode end or bound violation (higher is better)