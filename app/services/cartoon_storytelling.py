"""Persistent pending-object staging for the bounded cartoon templates.

One original paper marker moves between task, thought, tab, draft, note and
return slot. All state is resolved from the plan and absolute time; drawing a
frame never mutates the story. This is deliberately not a scene-graph framework.
"""
from dataclasses import dataclass
import math

from PIL import ImageDraw

from app.services.cartoon_choreography import stage_at


@dataclass(frozen=True)
class StoryState:
    object_id: str
    semantic_state: str
    representation: str
    x: float
    y: float
    loop_active: bool
    parked: bool


DESTINATIONS = {
    "stop_work": (540, 610),
    "mental_persistence": (540, 350),
    "browser_open_tab": (540, 315),
    "unfinished_email": (660, 475),
    "write_next_step": (520, 540),
    "task_parked": (835, 300),
    "mental_release": (835, 300),
}


def state_at(scenes, t):
    position = (540, 610)
    for index, scene in enumerate(scenes):
        destination = DESTINATIONS.get(scene.scene_template, position)
        if t < scene.end or index == len(scenes) - 1:
            progress = max(0, min(1, (t - scene.start) / scene.duration))
            amount = min(1, progress / 0.8)
            amount = amount * amount * (3 - 2 * amount)
            parked = scene.scene_template == "mental_release" or (
                scene.scene_template == "task_parked" and progress >= 0.8
            )
            released = scene.scene_template == "mental_release" and progress >= 0.65
            return StoryState(
                scene.continuity_object,
                scene.state_after if progress >= 0.8 else scene.state_before,
                scene.focus_object,
                position[0] + (destination[0] - position[0]) * amount,
                position[1] + (destination[1] - position[1]) * amount,
                bool(scene.continuity_object) and not released,
                parked,
            )
        position = destination
    raise ValueError("story requires a scene")


def draw_story(renderer, image, scene, state, progress, t):
    """Draw visible causal representations around the same traveling marker."""
    draw = ImageDraw.Draw(image)
    sx, sy, sc = renderer.sx, renderer.sy, renderer.sc
    palette = renderer.palette
    ink, paper, yellow = palette.card_ink, palette.card, palette.accent

    def box(bounds, fill=paper, outline=ink, radius=24, width=6):
        x1, y1, x2, y2 = bounds
        draw.rounded_rectangle((sx(x1), sy(y1), sx(x2), sy(y2)),
                               radius=sc(radius), fill=fill, outline=outline, width=sc(width))

    def line(points, color=ink, width=7):
        draw.line([(sx(x), sy(y)) for x, y in points], fill=color, width=sc(width))

    def circle(x, y, radius, fill, outline=None):
        draw.ellipse((sx(x-radius), sy(y-radius), sx(x+radius), sy(y+radius)),
                     fill=fill, outline=outline, width=sc(5))

    def arrow(a, b, color=yellow):
        line([a, b], color, 9)
        angle = math.atan2(b[1]-a[1], b[0]-a[0])
        for delta in (-0.55, 0.55):
            line([b, (b[0]-24*math.cos(angle+delta), b[1]-24*math.sin(angle+delta))], color, 8)

    def pending_marker(x, y, size=1):
        # Identity is never a completion tick: pink dot, gold spine, unfinished line.
        box((x-48*size, y-58*size, x+48*size, y+58*size), radius=10)
        line([(x-34*size,y-43*size),(x-34*size,y+43*size)],yellow,9)
        circle(x+20*size,y-30*size,9*size,palette.host)
        for dy, length in ((-6,45),(16,34)):
            line([(x-19*size,y+dy*size),(x+length*size/2,y+dy*size)],ink,5)
        line([(x-19*size,y+38*size),(x-3*size,y+38*size)],ink,5)
        circle(x+14*size,y+38*size,3*size,palette.red)

    template = scene.scene_template
    stage = stage_at(scene, t)
    # Every representation occupies the same shared demonstration space.
    if template == "stop_work":
        box((225,320,855,740), fill="#D5DFEA")
        box((250,345,830,685))
        for y,length in ((410,380),(458,340),(506,180)):
            line([(300,y),(300+length,y)],"#A6AAB6",12)
        line([(180,750),(900,750)],palette.muted_text,20)
        # A pen visibly leaves the unfinished page; its unfinished line remains.
        lift=min(1,progress*2)
        line([(605+150*lift,585+90*lift),(675+150*lift,510+90*lift)],yellow,15)
        if stage.name in {"action", "emphasis", "resolve"}:
            arrow((780,660),(920,660))
    elif template in {"mental_persistence", "mental_release"}:
        # Thought cloud connected to the viewer proxy, rather than an isolated label.
        if template != "mental_release" or progress<0.72:
            for x,y,r in ((420,350,145),(565,305,155),(685,375,125)):
                circle(x,y,r,"#DDEDF1")
            circle(710,640,28,"#DDEDF1")
            circle(740,750,16,"#DDEDF1")
            if state.loop_active and stage.name in {"establish", "action", "emphasis"}:
                angle=t*2.1
                draw.arc((sx(330),sy(190),sx(755),sy(590)), 20,335,
                         fill=palette.red,width=sc(10))
                circle(540+200*math.cos(angle),390+180*math.sin(angle),13,yellow)
            if template == "mental_release":
                # The mental tab shrinks shut while the actual item stays parked.
                width=max(0,230*(1-progress/0.72))
                if width>8:
                    box((540-width,330,540+width,440),fill="#BFC6DA")
        if template == "mental_release" and progress>=0.65:
            arrow((475,660),(240,660),palette.cyan)
            # A new focus (a sprouting plant) appears outside the former thought loop.
            line([(245,675),(245,520)],palette.green,10)
            draw.ellipse((sx(177),sy(530),sx(245),sy(580)),fill=palette.green)
            draw.ellipse((sx(245),sy(490),sx(310),sy(545)),fill=palette.green)
            box((207,675,285,735),fill=palette.host,radius=8)
    elif template == "browser_open_tab":
        box((165,250,915,760))
        for i in range(4):
            x=190+i*175
            active=i==2
            box((x,270,x+160,355),fill=yellow if active else "#BEC3D0",radius=12)
            if not active:
                line([(x+25,310),(x+130,310)],"#8E96A8",6)
        for y in (440,490,540,590):
            line([(220,y),(815,y)],"#C8CAD4",15)
        # Other tabs dim while the pending item stays open.
        if stage.name in {"emphasis", "resolve"}:
            for x in (235,410,760):
                line([(x,310),(x+45,310)],paper,15)
    elif template == "unfinished_email":
        box((190,250,890,770))
        # Envelope in the header establishes an email without relying on a label.
        box((235,285,350,365),radius=5)
        line([(235,285),(293,330),(350,285)],ink,5)
        line([(395,320),(800,320)],"#AFB6C5",10)
        for y,length in ((410,450),(465,350),(520,210)):
            line([(250,y),(250+length,y)],"#939CAB",12)
        if int(t*2)%2==0:
            line([(468,503),(468,537)],ink,5)
        # Unsent envelope and unfinished text persist; no send/completion icon.
        box((670,660,825,720),fill="#D4D7DE",radius=12)
    elif template in {"write_next_step","task_parked"}:
        box((245,325,725,745))
        # Record one next action, with a visible pen moving along the new line.
        length=300*min(1,progress*1.6) if template=="write_next_step" else 300
        line([(305,440),(305+length,440)],ink,9)
        if template=="write_next_step":
            line([(305+length,440),(370+length,365)],yellow,16)
        line([(305,490),(470,490)],"#A3AAB5",8)
        # A return arrow, not a checkmark, marks the purpose of the recorded note.
        draw.arc((sx(310),sy(570),sx(450),sy(690)),20,290,fill=palette.cyan,width=sc(9))
        arrow((330,580),(305,605),palette.cyan)

    if template in {"task_parked","mental_release"}:
        # The same parked object is visible during release, still unfinished.
        box((755,200,960,420),fill="#C7D9D7",radius=20)
        line([(770,365),(945,365)],palette.cyan,12)
        circle(923,228,23,paper,ink)
        line([(923,211),(923,228),(936,236)],ink,4)
        if template=="task_parked" and stage.name != "resolve":
            arrow((665,430),(760,340),palette.cyan)

    pending_marker(state.x,state.y)
    if state.loop_active and template not in {"mental_persistence", "mental_release", "stop_work"}:
        # Carry the unresolved mental loop through the metaphor and example,
        # rather than letting it disappear whenever the surrounding UI changes.
        draw.arc((sx(state.x-68),sy(state.y-78),sx(state.x+68),sy(state.y+78)),
                 25,335,fill=palette.red,width=sc(5))

    # The recurring original cast grounds the demonstration in a shared setting.
    for host,x in ((True,275),(False,790)):
        owns_voice=scene.narration_actor == ("host" if host else "guest")
        emotion="happy" if template=="mental_release" and progress>.65 else (
            "concerned" if state.loop_active and not host else scene.emotion)
        renderer._character(
            draw,x=sx(x),base_y=sy(1140),color=palette.host if host else palette.guest,
            shadow=palette.host_shadow if host else palette.guest_shadow,
            emotion=emotion,action="point" if host else "react",
            mouth=renderer.story_mouth(t,owns_voice),facing=1 if host else -1,
            t=t,local_progress=progress,scale=.88,
        )
