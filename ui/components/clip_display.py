# Functions for displaying clips in various formats
import base64
import io
import logging
import os

import streamlit as st
from matplotlib import pyplot as plt

# Configure logger for this module
logger = logging.getLogger(__name__)

# Constants for styling and display limits
MAX_TAGS_DISPLAY = 10
TIMELINE_PLOT_DPI = 150
DEFAULT_SCORE = 0
MAX_SCORE_FOR_COLOR = 100  # Assume scores (for color mapping) are within 0-100

# CSS Style Helpers
def _add_play_button_css():
    """Adds the CSS for play button overlay styling to the page."""
    st.markdown("""
    <style>
    div.stButton > button {
        position: relative;
        top: -150px;  /* Adjusted to better center the button vertically */
        background-color: rgba(0,0,0,0.7);  /* Increased opacity for better contrast */
        color: white;
        font-size: 24px;
        border-radius: 50%;
        width: 80px;  /* Increased clickable area */
        height: 80px;  /* Increased clickable area */
        display: flex;
        align-items: center;
        justify-content: center;
        border: none;
        z-index: 999;
    }
    div.stButton > button:hover {
        background-color: rgba(0,0,0,0.9);  /* Darker on hover for better feedback */
        transform: scale(1.1);  /* Slight grow effect on hover */
        transition: all 0.2s ease;
    }
    /* Make clickable area more apparent */
    div.stButton {
        text-align: center;
        margin-top: -100px;  /* Adjust container position */
    }
    </style>
    """, unsafe_allow_html=True)

def _format_score_text(clip: dict) -> str:
    """Formats the score display string including initial and AI scores."""
    score_text = f"Initial Score: {clip.get('score', DEFAULT_SCORE)}"
    ai_viral_score = clip.get("ai_viral_score")
    ai_monetization_score = clip.get("ai_monetization_score")

    # Collect AI score parts if they exist
    ai_score_parts = []
    if ai_viral_score is not None:
        ai_score_parts.append(f"Viral:{ai_viral_score:.0f}")
    if ai_monetization_score is not None:
        ai_score_parts.append(f"Monetization:{ai_monetization_score:.0f}")

    # Append AI scores to the main text if available
    if ai_score_parts:
        score_text += f" (AI {' '.join(ai_score_parts)})"

    return score_text


def _display_clip_details(clip: dict):
    """
    Helper function to display common details for a clip within a Streamlit container.

    Args:
        clip (dict): A dictionary containing clip metadata. Expected keys include
                     'score', 'ai_viral_score', 'ai_monetization_score',
                     'category', 'start', 'end', 'quip', 'colors', 'ai_tags',
                     'ai_recommendations'. Missing keys are handled gracefully.
    """
    # Display Formatted Score
    st.caption(_format_score_text(clip))

    # Display Category and Time Range
    st.caption(
        f"Category: {clip.get('category', 'N/A')} | "
        f"Time: {clip.get('start', 0):.1f}s - {clip.get('end', 0):.1f}s"
    )

    # Display Score Explanation
    score_explanation = clip.get("score_explanation")
    if score_explanation:
        st.markdown(f"**Score Explanation:** {score_explanation}")
    else:
        st.caption("**Score Explanation:** N/A")

    # Display AI Tags
    ai_tags = clip.get("ai_tags", [])
    # Validate that 'ai_tags' is a list of strings
    if isinstance(ai_tags, list) and all(isinstance(t, str) for t in ai_tags):
        if ai_tags:
            tags_display = ai_tags[:MAX_TAGS_DISPLAY]
            tag_html = " ".join(
                f"<span style='display:inline-block; background-color:#444; color: #eee; "
                f"padding: 1px 5px; border-radius:10px; margin-right:3px; margin-bottom:3px; "
                f"font-size:0.8em;'>{tag}</span>"
                for tag in tags_display
            )
            st.markdown(f"**AI Tags:** {tag_html}", unsafe_allow_html=True)
            if len(ai_tags) > MAX_TAGS_DISPLAY:
                st.caption(f"...and {len(ai_tags) - MAX_TAGS_DISPLAY} more tags.")
        else:
            st.caption("**AI Tags:** N/A (Empty list)")
    else:
        st.caption("**AI Tags:** N/A (Invalid data)")

    # Display Quip
    quip = clip.get("quip")
    if quip:
        st.markdown(f"**Quip:** *{quip}*")
    else:
        st.caption("**Quip:** N/A")

    # Display AI Recommendations in an expander
    ai_recs = clip.get("ai_recommendations", [])
    # Validate that 'ai_recs' is a list of strings
    if isinstance(ai_recs, list) and all(isinstance(r, str) for r in ai_recs):
        if ai_recs:
            # Removed key parameter from expander
            with st.expander("AI Recommendations", expanded=False):
                for rec in ai_recs:
                    st.markdown(f"- {rec}")
    # If no valid recommendations, the expander is simply not shown.


def display_clips_gallery(clips: list, cols: int = 4, show_details: bool = True):
    """
    Displays clips in a responsive gallery format using Streamlit columns.
    Includes clickable thumbnails that show/hide video players.

    Args:
        clips (list): A list of clip dictionaries. Each clip requires an 'id'.
        cols (int): The maximum number of columns for the gallery.
        show_details (bool): Whether to display clip details under the thumbnail.
    """
    if not clips:
        st.info("No clips to display in the gallery.")
        return

    # Ensure at least 1 column and not more columns than clips
    num_cols = min(len(clips), max(1, cols))
    gallery_cols = st.columns(num_cols)

    for i, clip in enumerate(clips):
        clip_id = clip.get("id")
        if not clip_id:
            logger.warning(f"Clip at index {i} is missing an 'id'. Skipping.")
            continue  # Need ID for unique keys

        # Determine the column for the current clip
        col_index = i % num_cols
        with gallery_cols[col_index]:
            # Use a container for a card-like appearance
            with st.container(border=True):
                # Video Player Logic (using session state to toggle visibility)
                clip_video_path = clip.get("clip_path")
                can_play = clip_video_path and os.path.exists(clip_video_path)

                # Session state key to manage the player visibility for this specific clip
                player_state_key = f"show_player_gal_{clip_id}"

                # Initialize state if it doesn't exist
                if player_state_key not in st.session_state:
                    st.session_state[player_state_key] = False
                
                # Thumbnail and player container - all in one
                thumb_path = clip.get("thumbnail")
                
                # Handle thumbnail display
                if thumb_path and os.path.exists(thumb_path):
                    try:
                        # If player is visible, show the video
                        if st.session_state[player_state_key] and can_play:
                            st.video(clip_video_path)
                        else:
                            # Add custom CSS for floating buttons
                            _add_play_button_css()

                            # Display thumbnail - removed key parameter
                            caption = f"Tag: {clip.get('tag', 'N/A')}" if not show_details and clip.get("tag") else None
                            st.image(
                                thumb_path,
                                use_container_width=True,
                                caption=caption
                            )
                            
                            # Add centered play button if video is available
                            if can_play:
                                # Use columns to center the button
                                cols = st.columns([2, 1, 2])
                                with cols[1]:
                                    if st.button("▶", key=f"play_btn_gal_{clip_id}"):
                                        # Toggle video visibility
                                        st.session_state[player_state_key] = not st.session_state[player_state_key]
                                        st.rerun()
                            else:
                                st.caption("⚠️ Video not available")
                    except Exception as e:
                        logger.warning(f"Error displaying thumbnail {thumb_path}: {e}")
                        st.caption("⚠️ Thumbnail Error")
                else:
                    # Placeholder if no thumbnail exists
                    tag_display = clip.get("tag", "No Tag")
                    st.caption("🖼️ No Thumbnail Available")
                    if not show_details:
                        st.caption(f"Tag: {tag_display}")
                    
                    # Still offer play button if video exists but no thumbnail
                    if can_play:
                        if st.button(
                            "▶️ Play Video",
                            key=f"no_thumb_button_gal_{clip_id}"
                        ):
                            st.session_state[player_state_key] = not st.session_state[player_state_key]
                            st.rerun()
                    
                    # Show video if state is active
                    if st.session_state[player_state_key] and can_play:
                        st.video(clip_video_path)
                
                # Display Details (optional)
                if show_details:
                    _display_clip_details(clip)


def display_clips_list(clips: list):
    """
    Displays clips in a list format, where each clip is within an expander.
    Thumbnails are clickable to show/hide video players.

    Args:
        clips (list): A list of clip dictionaries. Each clip requires an 'id'.
    """
    if not clips:
        st.info("No clips to display in the list.")
        return

    for i, clip in enumerate(clips):
        clip_id = clip.get("id")
        if not clip_id:
            logger.warning(f"Clip at index {i} in list is missing an 'id'. Skipping.")
            continue # Need ID for unique keys

        # Format the title for the expander including score and time range
        score_text = _format_score_text(clip).replace("Initial Score:", "Score:") # Shorter label
        time_range = f"{clip.get('start', 0):.1f}s - {clip.get('end', 0):.1f}s"
        tag = clip.get('tag', 'No Tag')
        expander_title = f"[{score_text}] {tag} ({time_range})"

        # Create expander (removed key parameter)
        with st.expander(expander_title):
            # Use columns for layout: Thumbnail | Details
            list_cols = st.columns([1, 3])

            # Video path and state management
            clip_video_path = clip.get("clip_path")
            can_play = clip_video_path and os.path.exists(clip_video_path)
            
            # Session state key for this clip's player visibility
            player_state_key = f"show_player_list_{clip_id}"
            
            # Initialize state if it doesn't exist
            if player_state_key not in st.session_state:
                st.session_state[player_state_key] = False

            # Column 1: Thumbnail or Video
            with list_cols[0]:
                thumb_path = clip.get("thumbnail")
                
                # If video player is active, show it instead of thumbnail
                if st.session_state[player_state_key] and can_play:
                    st.video(clip_video_path)
                else:
                    # Otherwise show the thumbnail with play button overlay
                    if thumb_path and os.path.exists(thumb_path):
                        try:
                            # Add custom CSS for floating buttons
                            _add_play_button_css()
                            
                            # Display the thumbnail - removed key parameter
                            st.image(
                                thumb_path,
                                use_container_width=True
                            )
                            
                            # Add clickable button overlay if video is available
                            if can_play:
                                # Use columns to center the button
                                cols = st.columns([2, 1, 2])
                                with cols[1]:
                                    if st.button("▶", key=f"play_btn_list_{clip_id}"):
                                        # Toggle video visibility
                                        st.session_state[player_state_key] = not st.session_state[player_state_key]
                                        st.rerun()
                        except Exception as e:
                            logger.warning(
                                f"Error displaying thumbnail {thumb_path} in list: {e}"
                            )
                            st.caption("⚠️ Thumbnail Error")
                    else:
                        st.caption("🖼️ No Thumbnail")
                        if can_play:
                            if st.button(
                                "▶️ Play Video",
                                key=f"no_thumb_list_button_{clip_id}"
                            ):
                                st.session_state[player_state_key] = not st.session_state[player_state_key]
                                st.rerun()

            # Column 2: Clip Details
            with list_cols[1]:
                _display_clip_details(clip)  # Reuse the helper function

            # Show message if video can't be played
            if st.session_state[player_state_key] and not can_play:
                st.warning("Video file seems to be missing.")
                st.session_state[player_state_key] = False  # Reset state
            elif not can_play:
                st.caption("ℹ️ Video clip file not found or path is missing.")

            st.markdown("---") # Separator within the expander


def display_clips_timeline(clips: list, video_duration: float, sort_by: str):
    """
    Displays clips on a Matplotlib timeline visualization, embedded as an image.
    Includes an optional expander below the timeline to show details and players.

    Args:
        clips (list): List of clip dictionaries, pre-sorted as desired.
                      Requires 'id', 'start', 'end', 'tag', and score fields.
        video_duration (float): The total duration of the source video in seconds.
        sort_by (str): A string indicating how the clips were sorted (used for labeling).
    """
    if not clips or not video_duration or video_duration <= 0:
        st.info("Not enough data to display timeline (requires clips and video duration).")
        return

    st.subheader("Video Timeline Visualization")

    try:
        # --- Plot Configuration ---
        # Dynamic figure height based on the number of clips
        fig_height = max(4, len(clips) * 0.4)
        fig, ax = plt.subplots(figsize=(14, fig_height), facecolor="#2E2E2E")
        plt.style.use("seaborn-v0_8-darkgrid") # Use a visually appealing style
        ax.set_facecolor("#3B3B3B")

        # Determine which score to use for coloring (prioritize AI scores)
        has_ai_viral = any(c.get("ai_viral_score") is not None for c in clips)
        has_ai_monetization = any(c.get("ai_monetization_score") is not None for c in clips)

        y_labels = [] # Labels for the Y-axis

        # --- Plotting Each Clip ---
        for i, clip in enumerate(clips):
            start = clip.get("start", 0)
            end = clip.get("end", 0)
            duration = max(0, end - start) # Ensure duration is non-negative

            # Determine the score value used for coloring the bar
            score_for_color = clip.get("score", DEFAULT_SCORE) # Fallback to initial score
            if has_ai_viral:
                score_for_color = clip.get("ai_viral_score", DEFAULT_SCORE)
            elif has_ai_monetization:
                score_for_color = clip.get("ai_monetization_score", DEFAULT_SCORE)

            # Normalize score to 0-1 range for the colormap
            # Clamp score between 0 and MAX_SCORE_FOR_COLOR before normalizing
            normalized_score = max(0, min(MAX_SCORE_FOR_COLOR, score_for_color)) / MAX_SCORE_FOR_COLOR
            bar_color = plt.cm.coolwarm(normalized_score) # Apply colormap (cool=low, warm=high)

            # Draw the horizontal bar for the clip
            ax.barh(
                y=i,           # Y position based on sorted index
                width=duration,# Bar length
                left=start,    # Bar start position
                height=0.6,    # Bar thickness
                color=bar_color,
                edgecolor="white",
                linewidth=0.5,
            )

            # --- Text Label on the Bar ---
            # Shorten tag if necessary
            tag = clip.get('tag', 'Clip')
            label_tag = (f"{tag[:20]}..." if len(tag) > 20 else tag)

            # Format score text for the label (concise version)
            score_parts = [f"I:{clip.get('score', DEFAULT_SCORE)}"]
            if clip.get("ai_viral_score") is not None:
                score_parts.append(f"V:{clip['ai_viral_score']:.0f}")
            if clip.get("ai_monetization_score") is not None:
                score_parts.append(f"M:{clip['ai_monetization_score']:.0f}")
            score_label = f" ({' '.join(score_parts)})"
            full_label_text = label_tag + score_label

            # Calculate text color based on bar color luminance for contrast
            luminance = (0.299 * bar_color[0] + 0.587 * bar_color[1] + 0.114 * bar_color[2])
            text_color = "white" if luminance < 0.5 else "black"

            # Add the text label centered on the bar
            ax.text(
                x=start + duration / 2,
                y=i,
                s=full_label_text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
                weight="bold",
                clip_on=True, # Prevent text spilling outside plot area
            )

            # Prepare Y-axis labels (Clip index and truncated tag)
            y_labels.append(f"#{i+1} ({label_tag[:15]})") # Even shorter tag for axis

        # --- Axis and Plot Styling ---
        ax.set_xlim(0, video_duration)
        ax.set_ylim(-0.5, len(clips) - 0.5) # Adjust Y limits to fit bars nicely
        ax.set_yticks(range(len(clips)))
        ax.set_yticklabels(y_labels, color="white", fontsize=9)
        ax.invert_yaxis() # Show clip #1 at the top

        ax.set_xlabel("Time (seconds)", color="white")
        ax.set_ylabel(f"Clips (Sorted by {sort_by})", color="white")
        ax.set_title("Clip Timeline", color="white")
        ax.tick_params(colors="white", which="both") # Style ticks
        ax.grid(True, axis="x", linestyle=":", alpha=0.4, color="gray") # Faint vertical grid

        # Style the plot borders (spines)
        for spine in ax.spines.values():
            spine.set_color("white")

        # --- Color Bar Legend ---
        # Create a scalar mappable for the color bar
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.coolwarm,
            norm=plt.Normalize(vmin=0, vmax=MAX_SCORE_FOR_COLOR)
        )
        sm.set_array([]) # Required dummy array
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, aspect=30)
        cbar.set_label("Score (for color)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white") # Colorbar ticks
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white") # Colorbar labels

        # --- Save Plot to Buffer and Display ---
        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to prevent overlap
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=TIMELINE_PLOT_DPI, facecolor=fig.get_facecolor())
        buf.seek(0)
        # Encode image to base64 string for embedding in Streamlit
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig) # Close the plot to free memory

        st.image(f"data:image/png;base64,{img_str}", use_container_width=True)

        # --- Optional Expander for Clip Details Below Timeline ---
        with st.expander("Show Clip Details & Player for Timeline Clips"):
            if not clips:
                st.write("No clips were included in the timeline.")
            else:
                for i, clip in enumerate(clips): # Iterate through the *same* sorted clips
                    clip_id = clip.get("id")
                    if not clip_id:
                        logger.warning(f"Clip #{i+1} in timeline details missing 'id'.")
                        st.caption(f"Clip #{i+1}: ID missing, cannot display details fully.")
                        continue

                    # Create a title for this clip's section
                    score_text_detail = _format_score_text(clip).replace("Initial Score:", "Score:")
                    time_range_detail = f"{clip.get('start',0):.1f}s - {clip.get('end',0):.1f}s"
                    tag_detail = clip.get('tag', 'N/A')
                    detail_title = f"Clip #{i+1}: {tag_detail} ({score_text_detail}, Time: {time_range_detail})"

                    # Create inner expander (removed key parameter)
                    with st.expander(detail_title):
                        # Video path and state management
                        clip_video_path = clip.get("clip_path")
                        can_play = clip_video_path and os.path.exists(clip_video_path)
                        
                        # Session state key for this clip's player visibility
                        player_state_key = f"show_player_timeline_{clip_id}"
                        
                        # Initialize state if it doesn't exist
                        if player_state_key not in st.session_state:
                            st.session_state[player_state_key] = False
                        
                        # If player is active, show the video
                        if st.session_state[player_state_key] and can_play:
                            st.video(clip_video_path)
                        else:
                            # Display thumbnail with play button overlay
                            thumb_path = clip.get("thumbnail")
                            if thumb_path and os.path.exists(thumb_path):
                                try:
                                    # Add custom CSS for floating buttons
                                    _add_play_button_css()
                                    
                                    # Display the thumbnail (removed key parameter)
                                    st.image(thumb_path, use_container_width=True)
                                    
                                    if can_play:
                                        # Use columns to center the button
                                        cols = st.columns([2, 1, 2])
                                        with cols[1]:
                                            if st.button("▶", key=f"play_btn_timeline_{clip_id}"):
                                                # Toggle video visibility
                                                st.session_state[player_state_key] = not st.session_state[player_state_key]
                                                st.rerun()
                                except Exception as e:
                                    logger.error(f"Failed to display thumbnail for timeline clip: {e}")
                                    st.info("ℹ️ Thumbnail not available")
                            else:
                                st.info("ℹ️ Thumbnail not available")
                                if can_play:
                                    if st.button(
                                        "▶️ Play Video",
                                        key=f"timeline_no_thumb_button_{clip_id}"
                                    ):
                                        st.session_state[player_state_key] = not st.session_state[player_state_key]
                                        st.rerun()
                        
                        # Show warning if video is suddenly unavailable
                        if st.session_state[player_state_key] and not can_play:
                            st.warning("Video file seems to be missing.")
                            st.session_state[player_state_key] = False  # Reset state
                        elif not can_play:
                            st.info("ℹ️ Video clip file not available for this segment.")

                        # Display the details using the helper
                        _display_clip_details(clip)

                    st.markdown("---") # Separator between clip detail sections

    except Exception as plot_err:
        logger.error(f"Failed to generate timeline plot: {plot_err}", exc_info=True)
        st.warning(f"⚠️ Could not display timeline visualization: {plot_err}")