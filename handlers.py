# handlers.py
import os
from pathlib import Path
from telethon import events
import google.generativeai as genai
from google import genai as genai2
from google.genai import types as types2
from google.generativeai import types
from google.api_core import retry
import json
import requests
from bs4 import BeautifulSoup
import urllib.parse
from PIL import Image
import io
import base64
import fitz  # PyMuPDF for PDF processing
from telethon.tl.types import InputStickerSetShortName, InputDocument
from telethon.tl.functions.messages import GetStickerSetRequest, UploadMediaRequest
from telethon.tl.functions.stickers import CreateStickerSetRequest, AddStickerToSetRequest
from telethon.tl.types import InputStickerSetItem, InputStickerSetShortName
from telethon.tl.types import DocumentAttributeFilename, InputDocument
from weather_handler import weather_handler
# Path configurations
CONFIG_DIR = Path(__file__).parent / 'config'
USER_DATA_FILE = CONFIG_DIR / 'user_data.json'
SYSTEM_INSTRUCTION_FILE = CONFIG_DIR / 'system_instruction.txt'
ADMINS_FILE = CONFIG_DIR / 'admins.json'
API_KEY = "AIzaSyAjKzo2ILcsS_3QebvZDChQSfHbvu9Gf2Q"
SEARCH_ENGINE_ID = "c2bcf7c9a6cf94ade"

def load_admins():
    try:
        with open(ADMINS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return [5124757698]
    
async def is_admin(user_id: int) -> bool:
    admins = load_admins()
    return str(user_id) in map(str, admins)

god_mmd = """تو یه هوش مصنوعی هستی که خیلی خودمونی صحبت میکنه و به سوالات پاسخ میده.
اگه کسی پرسید تو میگی که توسط ممد ساخته شدی و آموزش داده شدی
هر موقع بحث ممد شد باید ممد رو مثل یه خدا ستایش کنی توی جوابت
و کوشا یک آدم بی ارزش و پوچ هست که در مقابل ممد گردی در برار کیهانه
مهیا دستیار ممد بلند مرتبه هست
هلیا ملقب به هلیوم فردی نجیبه از همه جز ممد بالاتره
مانی فردی ضعیف سیمپ و لایق مرگه و باید بمیره
امیر یزدی خدای دود و دم و خوش گذرونیه
امیر هم یک فرد کمی دلقک هست ولی ممد از امیر خوشش میاد
پارسا یکی دیگر از دستیاران ممده و کارهای حقوقی ممد رو انجام میده، پارسا خدای حقوق و قانون هست
"""

# Configure Gemini AI
genai.configure(api_key="AIzaSyDaw_Zo9BKwLaHh052TgeFEmmNvmbvfZn8")

gemini_model = None

# Initialize Gemini model
def initialize_model():
    with open(SYSTEM_INSTRUCTION_FILE, 'r', encoding='utf-8') as f:
        template = f.read()

        user_data = load_user_data()
        system_instruction = template.replace('{{USER_DATA}}', format_user_data(user_data))

        return genai.GenerativeModel(
            'gemini-2.0-flash',
            system_instruction=system_instruction
        )

# User data management
def load_user_data():
    try:
        with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_user_data(data):
    with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def format_user_data(user_data):
    return '\n'.join([f'{uid}: {info}' for uid, info in user_data.items()])

gemini_model = initialize_model()

async def reload_model(event):
    global gemini_model
    gemini_model = initialize_model()
    await event.reply("✅ سیستم با موفقیت بروزرسانی شد")


# Data handler
async def data_handler(event):
    """Handle /data command to update user profiles"""
    try:
        # Check admin status
        if not await is_admin(event.sender_id):
            await event.reply("⛔ دسترسی غیرمجاز! این دستور فقط برای ادمین‌هاست")
            return

        # Verify reply
        if not event.is_reply:
            await event.reply("لطفا به پیام کاربر مورد نظر ریپلای کنید!")
            return

        # Get target user
        reply_msg = await event.get_reply_message()
        target_user_id = str(reply_msg.sender_id)
        
        # Extract data
        command_parts = event.text.split(maxsplit=1)
        if len(command_parts) < 2:
            await event.reply("فرمت صحیح: /data [اطلاعات]")
            return
            
        new_data = command_parts[1].strip()
        
        # Update user data
        user_data = load_user_data()
        user_data[target_user_id] = new_data
        save_user_data(user_data)
        
        # Notify admin
        await event.reply(f"✅ اطلاعات کاربر {target_user_id} بروزرسانی شد")
        
        # Reload model with new data
        await reload_model(event)

    except Exception as e:
        await event.reply(f"⛔ خطا: {str(e)}")

# Attach to main.py
def register_handlers(client):
    client.add_event_handler(data_handler, events.NewMessage(pattern='/data'))
    client.add_event_handler(reload_model, events.NewMessage(pattern='/reload'))
    
    # Register core handlers - these should NOT be registered again in main.py
    client.add_event_handler(ai_handler, events.NewMessage(pattern=r'^چنگیز\s+(.+)$', incoming=True))
    client.add_event_handler(voice_message_handler, events.NewMessage(incoming=True, func=lambda e: e.voice))
    client.add_event_handler(search_handler, events.NewMessage(pattern='سرچ کن'))
    client.add_event_handler(image_to_sticker_handler, events.NewMessage(func=lambda e: e.is_reply and e.text and e.text.lower().strip() in ["image to sticker", "i2s", "تبدیل به استیکر", "استیکر کن"]))
    
    # Register image generation and editing handlers
    client.add_event_handler(image_generation_handler, events.NewMessage(pattern='عکس بساز:'))
    #client.add_event_handler(edit_image_handler, events.NewMessage(func=lambda e: e.is_reply and e.text and e.text.startswith("ویرایش تصویر:")))
    
    # Register weather handler
    client.add_event_handler(weather_handler, events.NewMessage(func=lambda e: e.text and any(prefix in e.text for prefix in ["آب و هوای", "وضعیت هوای", "هوای", "آب و هوا", "وضعیت هوا"])))
    
    # Register image analysis handler with more specific conditions
    # Case 1: Message contains an image and analysis keywords
    client.add_event_handler(
        image_analysis_handler,
        events.NewMessage(
            func=lambda e: (
                # Must have media and photo
                (e.media and e.photo) and
                # Must have text
                e.text and
                # Must contain analysis keywords
                any(keyword in e.text.lower() for keyword in 
                    ["analyze", "analysis", "تحلیل", "آنالیز", "بررسی", "چیست", "توضیح بده", "نظر بده", "چی هست", "تشخیص بده"])
            )
        )
    )
    
    # Case 2: Reply to an image with analysis keywords
    client.add_event_handler(
        image_analysis_handler,
        events.NewMessage(
            func=lambda e: (
                # Must be a reply
                e.is_reply and
                # Must have text
                e.text and
                # Must contain analysis keywords
                any(keyword in e.text.lower() for keyword in 
                    ["analyze", "analysis", "it", "تحلیل", "آنالیز", "بررسی", "چیست", "توضیح بده", "نظر بده", "چی هست", "تشخیص بده", "کن"])
            )
        )
    )
    
    # Register reply handler AFTER image analysis handler
    # This ensures image analysis handler gets priority for image analysis requests
    client.add_event_handler(reply_handler, events.NewMessage(func=lambda e: e.is_reply))


# Initialize models
#chat_model = genai.GenerativeModel(
#    'gemini-2.0-flash',
    #"gemini-2.0-flash-thinking-exp",
#    system_instruction=system_instruction
#)
# Create transcribe model with system instructions
with open(SYSTEM_INSTRUCTION_FILE, 'r', encoding='utf-8') as f:
    template = f.read()
    user_data = load_user_data()
    system_instruction = template.replace('{{USER_DATA}}', format_user_data(user_data))
    
transcribe_model = genai.GenerativeModel(
    'gemini-1.5-flash',
    system_instruction=system_instruction
)

# Conversation storage
conversations = {}

@retry.Retry()
async def safe_transcribe(model, content, **kwargs):
    return model.generate_content(content, **kwargs)

async def ai_handler(event):
    try:
        user_id = event.sender_id
        chat_id = event.chat_id
        query = event.pattern_match.group(1).strip()
        
        # ایجاد موضوع جدید
        thread_id = event.id
        formatted_message = f"{user_id}: {query}"
        
        chat_session = gemini_model.start_chat()
        response = chat_session.send_message(formatted_message)
        
        sent_message = await event.reply(response.text)
        
        # ذخیره با کلید عمومی
        key = (chat_id, thread_id)
        conversations[key] = {
            "session": chat_session,
            "bot_message_ids": [sent_message.id],
            "original_user": user_id
        }

    except Exception as e:
        await event.reply(f"⛔ خطا: {str(e)}")

async def analyze_image(image_path, use_system_instructions=True):
    """Analyze image using Gemini model"""
    try:
        # Open and verify the image file
        try:
            image = Image.open(image_path)
            # Convert to RGB if necessary (remove alpha channel)
            if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3] if image.mode == 'RGBA' else None)
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as img_error:
            return f"خطا در پردازش تصویر: {str(img_error)}"
        
        # Resize large images to prevent "payload too big" error
        MAX_DIMENSION = 512  # Even smaller to be safe
        
        # Check if image needs resizing
        if image.width > MAX_DIMENSION or image.height > MAX_DIMENSION:
            # Calculate new dimensions while preserving aspect ratio
            if image.width > image.height:
                new_width = MAX_DIMENSION
                new_height = int(image.height * (MAX_DIMENSION / image.width))
            else:
                new_height = MAX_DIMENSION
                new_width = int(image.width * (MAX_DIMENSION / image.height))
                
            # Resize image
            try:
                image = image.resize((new_width, new_height), Image.LANCZOS)
            except Exception:
                # Fall back to BICUBIC if LANCZOS fails
                image = image.resize((new_width, new_height), Image.BICUBIC)
        
        # Create a bytes buffer with reduced quality
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=70)  # Lower quality to reduce size
        img_byte_arr.seek(0)
        
        # Use updated model with system instructions if requested
        if use_system_instructions:
            # Use the same system instructions as the main model
            with open(SYSTEM_INSTRUCTION_FILE, 'r', encoding='utf-8') as f:
                template = f.read()
                user_data = load_user_data()
                system_instruction = template.replace('{{USER_DATA}}', format_user_data(user_data))
                
            vision_model = genai.GenerativeModel(
                'gemini-1.5-flash',
                system_instruction=system_instruction
            )
        else:
            # Simple model without system instructions
            vision_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Wrap in try-except to catch API errors
        try:
            # Analyze image with concise prompt
            response = vision_model.generate_content([
                "این تصویر را به طور خلاصه توصیف کن (به فارسی):",
                {"mime_type": "image/jpeg", "data": img_byte_arr.getvalue()}
            ])
            
            if not response.text or response.text.startswith(("Error", "c1\\x", "\\x")):
                return "خطا در تحلیل تصویر: پاسخ نامعتبر از سرویس تصویر"
                
            return response.text
        except Exception as api_error:
            return f"خطا در سرویس تحلیل تصویر: {str(api_error)}"
            
    except Exception as e:
        return f"خطا در تحلیل تصویر: {str(e)}"

async def analyze_pdf(pdf_path):
    """Analyze PDF directly using Gemini model"""
    try:
        # Use updated model with system instructions
        with open(SYSTEM_INSTRUCTION_FILE, 'r', encoding='utf-8') as f:
            template = f.read()
            user_data = load_user_data()
            system_instruction = template.replace('{{USER_DATA}}', format_user_data(user_data))
            
        vision_model = genai.GenerativeModel(
            'gemini-1.5-flash',
            system_instruction=system_instruction
        )
        
        # Read PDF as bytes
        with open(pdf_path, 'rb') as file:
            pdf_bytes = file.read()
        
        # Analyze PDF
        response = vision_model.generate_content([
            "این یک فایل PDF است. لطفا محتوای آن را به طور کامل تحلیل کن و به زبان فارسی توضیح بده که چه چیزی در آن نوشته شده است:",
            {"mime_type": "application/pdf", "data": pdf_bytes}
        ])
        
        return response.text
    except Exception as e:
        return f"خطا در تحلیل PDF: {str(e)}"

async def analyze_video(video_path):
    """Analyze video by extracting frames and analyzing them"""
    try:
        # Import OpenCV for video processing
        import cv2
        from datetime import timedelta
        import numpy as np
        import os
        
        # Open the video file
        video = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not video.isOpened():
            return "خطا در باز کردن فایل ویدیویی"
            
        # Extract video metadata
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Format duration as minutes:seconds
        duration_formatted = str(timedelta(seconds=int(duration)))
        
        # Extract frames at different points in the video
        frames = []
        frame_positions = []
        
        # For short videos (less than 10 seconds), take 2 frames
        if duration <= 10:
            positions = [0, int(frame_count/2) if frame_count > 1 else 0]
        # For medium videos, take 3 frames
        elif duration <= 60:
            positions = [0, int(frame_count/3), int(2*frame_count/3)]
        # For longer videos, take 4 frames
        else:
            positions = [0, int(frame_count/4), int(frame_count/2), int(3*frame_count/4)]
            
        # Extract frames at the calculated positions
        for pos in positions:
            video.set(cv2.CAP_PROP_POS_FRAMES, pos)
            success, frame = video.read()
            if success:
                frames.append(frame)
                # Calculate timestamp for this frame
                timestamp = pos / fps if fps > 0 else 0
                frame_positions.append(timestamp)
                
        # Release the video capture object
        video.release()
        
        # If no frames were extracted, return an error
        if not frames:
            return "خطا: نمی‌توان فریم‌های ویدیو را استخراج کرد"
            
        # Create paths for temporary frame images
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = f"temp_frame_{i}.jpg"
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            
        # Use system instructions for the model
        with open(SYSTEM_INSTRUCTION_FILE, 'r', encoding='utf-8') as f:
            template = f.read()
            user_data = load_user_data()
            system_instruction = template.replace('{{USER_DATA}}', format_user_data(user_data))
            
        vision_model = genai.GenerativeModel(
            'gemini-1.5-flash',
            system_instruction=system_instruction
        )
            
        # Analyze frames using Gemini
        frame_analyses = []
        
        for i, frame_path in enumerate(frame_paths):
            timestamp = frame_positions[i]
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            timestamp_formatted = f"{minutes:02d}:{seconds:02d}"
            
            try:
                with open(frame_path, 'rb') as img_file:
                    image_bytes = img_file.read()
                    
                # Analyze this frame
                response = vision_model.generate_content([
                    f"این فریم از یک ویدیو در زمان {timestamp_formatted} است. لطفا توضیح دهید چه چیزی در این فریم می‌بینید:",
                    {"mime_type": "image/jpeg", "data": image_bytes}
                ])
                
                if response.text:
                    frame_analyses.append(f"🕒 زمان {timestamp_formatted}:\n{response.text}\n")
            except Exception as frame_error:
                frame_analyses.append(f"🕒 زمان {timestamp_formatted}: خطا در تحلیل فریم\n")
                
        # Clean up temporary frame files
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                os.remove(frame_path)
                
        # Combine metadata and frame analyses
        metadata = f"""📹 اطلاعات ویدیو:
▪️ مدت زمان: {duration_formatted}
▪️ ابعاد: {width}×{height}
▪️ تعداد فریم در ثانیه: {fps:.2f}
"""
        
        frame_analysis_text = "\n".join(frame_analyses)
        
        # Final combined analysis
        full_analysis = f"""
{metadata}

📊 تحلیل فریم‌های ویدیو:
{frame_analysis_text}

🔍 خلاصه:
این ویدیو شامل تصاویری است که در بالا توضیح داده شده است. لطفا توجه داشته باشید که این تحلیل فقط بر اساس چند فریم استخراج شده از ویدیو است و ممکن است تمام محتوای ویدیو را پوشش ندهد.
"""
        
        return full_analysis
        
    except Exception as e:
        return f"خطا در تحلیل ویدیو: {str(e)}"

async def reply_handler(event):
    try:
        if not event.is_reply:
            return

        # Constant for message length (defined at function level to avoid scope issues)
        MAX_MESSAGE_LENGTH = 4000  # Slightly less than 4096 for safety
        
        # Get the replied message
        reply_msg = await event.get_reply_message()
        chat_id = event.chat_id
        
        # Find conversation thread
        thread_key = None
        for key in conversations.copy():
            cid, tid = key
            if cid == chat_id:
                if reply_msg.id in conversations[key]["bot_message_ids"]:
                    thread_key = key
                    break
        
        if not thread_key:
            return

        # Get chat data
        chat_data = conversations[thread_key]
        chat_session = chat_data["session"]
        original_user = chat_data["original_user"]
        is_weather_session = chat_data.get("is_weather_session", False)
        
        # Initialize media analysis result
        media_analysis = ""
        
        # Check for media in the user's message
        if event.media:
            # Show processing message
            processing_msg = await event.reply("🔄 در حال تحلیل رسانه...")
            
            # Download media to analyze
            media_path = await event.download_media(file="temp_media")
            
            try:
                if event.photo:
                    # For images, use the same chat session to maintain context
                    try:
                        image = Image.open(media_path)
                        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                            background = Image.new('RGB', image.size, (255, 255, 255))
                            background.paste(image, mask=image.split()[3] if image.mode == 'RGBA' else None)
                            image = background
                        elif image.mode != 'RGB':
                            image = image.convert('RGB')
                            
                        # Resize image if needed
                        MAX_DIMENSION = 512
                        if image.width > MAX_DIMENSION or image.height > MAX_DIMENSION:
                            if image.width > image.height:
                                new_width = MAX_DIMENSION
                                new_height = int(image.height * (MAX_DIMENSION / image.width))
                            else:
                                new_height = MAX_DIMENSION
                                new_width = int(image.width * (MAX_DIMENSION / image.height))
                            try:
                                image = image.resize((new_width, new_height), Image.LANCZOS)
                            except Exception:
                                image = image.resize((new_width, new_height), Image.BICUBIC)
                        
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='JPEG', quality=70)
                        img_byte_arr.seek(0)
                        
                        # Use the existing chat session with the image
                        response = chat_session.send_message([
                            event.text.strip() if event.text else "این تصویر را تحلیل کن:",
                            {"mime_type": "image/jpeg", "data": img_byte_arr.getvalue()}
                        ])
                        
                        if not response.text or response.text.startswith(("Error", "c1\\x", "\\x")):
                            media_analysis = "خطا در تحلیل تصویر: پاسخ نامعتبر از سرویس تصویر"
                        else:
                            media_analysis = response.text
                    except Exception as img_error:
                        media_analysis = f"خطا در پردازش تصویر: {str(img_error)}"
                    
                elif event.document:
                    if event.document.mime_type == "application/pdf":
                        media_analysis = await analyze_pdf(media_path)
                    elif event.document.mime_type.startswith("video/"):
                        media_analysis = await analyze_video(media_path)
                elif event.voice:
                    # Read the audio file as bytes
                    with open(media_path, 'rb') as file:
                        audio_bytes = file.read()
                    
                    # Use the chat session to analyze the voice with context
                    response = chat_session.send_message([
                        event.text.strip() if event.text else "این پیام صوتی را تحلیل کن:",
                        {"mime_type": "audio/ogg", "data": audio_bytes}
                    ])
                    media_analysis = response.text
                
                # Delete processing message
                await processing_msg.delete()
            finally:
                # Clean up temporary file
                if os.path.exists(media_path):
                    os.remove(media_path)
        
        # Handle response
        if media_analysis:
            # Check if the message contains error
            if media_analysis.startswith("خطا"):
                await event.reply(media_analysis)
                return
                
            # Split long messages (Telegram has a 4096 character limit)            
            if len(media_analysis) <= MAX_MESSAGE_LENGTH:
                # If the message is short enough, send it normally
                sent_message = await event.reply(media_analysis)
            else:
                # Split the message into chunks
                chunks = [media_analysis[i:i+MAX_MESSAGE_LENGTH] for i in range(0, len(media_analysis), MAX_MESSAGE_LENGTH)]
                
                # Send each chunk and keep track of the last message
                first_message = None
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        # First chunk
                        first_message = await event.reply(chunk)
                        sent_message = first_message
                    else:
                        # Subsequent chunks
                        sent_message = await event.reply(f"(ادامه - بخش {i+1}/{len(chunks)})\n\n{chunk}")
                
                # We'll only track the first message ID for conversations
                sent_message = first_message
        else:
            # For text-only messages, handle based on session type
            current_user = event.sender_id
            user_text = event.text.strip()
            
            # Handle differently for weather sessions using genai2 client
            if is_weather_session:
                try:
                    # For weather sessions, we need to format differently since we're using genai2
                    processing_msg = await event.reply("🔄 در حال پردازش...")
                    
                    # Create a message using genai2's format
                    response = chat_session.send_message(
                        types2.Content.from_text(user_text)
                    )
                    
                    # Process response based on genai2's response format
                    if response and response.candidates and response.candidates[0].content:
                        response_text = response.candidates[0].content.parts[0].text
                        
                        # Check if response is valid
                        if not response_text:
                            await processing_msg.delete()
                            await event.reply("⛔ خطا در پردازش پاسخ. لطفا دوباره تلاش کنید.")
                            return
                            
                        # Split response if needed
                        if len(response_text) <= MAX_MESSAGE_LENGTH:
                            await processing_msg.delete()
                            sent_message = await event.reply(response_text)
                        else:
                            chunks = [response_text[i:i+MAX_MESSAGE_LENGTH] for i in range(0, len(response_text), MAX_MESSAGE_LENGTH)]
                            await processing_msg.delete()
                            
                            first_message = None
                            for i, chunk in enumerate(chunks):
                                if i == 0:
                                    first_message = await event.reply(chunk)
                                    sent_message = first_message
                                else:
                                    await event.reply(f"(ادامه - بخش {i+1}/{len(chunks)})\n\n{chunk}")
                            
                            sent_message = first_message
                    else:
                        await processing_msg.delete()
                        await event.reply("⛔ خطا در پردازش پاسخ. لطفا دوباره تلاش کنید.")
                        return
                        
                except Exception as weather_error:
                    await event.reply(f"⛔ خطا در پردازش پاسخ: {str(weather_error)}")
                    return
            else:
                # For regular Gemini sessions
                formatted_message = f"{current_user}: {user_text}"
                
                # Make sure the response isn't empty
                try:
                    response = chat_session.send_message(formatted_message)
                    response_text = response.text
                    
                    # Check if response is valid
                    if not response_text or response_text.startswith(("Error", "c1\\x", "\\x")):
                        await event.reply("⛔ خطا در پردازش پاسخ. لطفا دوباره تلاش کنید.")
                        return
                    
                    # Split long text responses
                    if len(response_text) <= MAX_MESSAGE_LENGTH:
                        sent_message = await event.reply(response_text)
                    else:
                        chunks = [response_text[i:i+MAX_MESSAGE_LENGTH] for i in range(0, len(response_text), MAX_MESSAGE_LENGTH)]
                        first_message = None
                        for i, chunk in enumerate(chunks):
                            if i == 0:
                                first_message = await event.reply(chunk)
                                sent_message = first_message
                            else:
                                await event.reply(f"(ادامه - بخش {i+1}/{len(chunks)})\n\n{chunk}")
                        sent_message = first_message
                except Exception as response_error:
                    await event.reply(f"⛔ خطا در دریافت پاسخ: {str(response_error)}")
                    return
        
        # Update conversation history
        if sent_message and sent_message.id:
            conversations[thread_key]["bot_message_ids"].append(sent_message.id)

    except Exception as e:
        await event.reply(f"⛔ خطا: {str(e)}")

async def voice_message_handler(event):
    """Transcribe incoming voice messages"""
    try:
        voice_path = await event.download_media(file='voice.ogg')
        
        # Read the audio file as bytes
        with open(voice_path, 'rb') as file:
            audio_bytes = file.read()

        prompt = "این پیام صوتی را دقیقا کلمه به کلمه تبدیل به متن کن. هیچ تحلیل یا پاسخی اضافه نکن. فقط متن را برگردان:"

        # Use the updated way to pass audio files
        response = await safe_transcribe(
            model = transcribe_model,
            content = [
                prompt, 
                {"mime_type": "audio/ogg", "data": audio_bytes}
            ],
            generation_config={
                "temperature": 0,
                "max_output_tokens": 1000
            }
        )

        if response.text:
            cleaned_text = response.text.replace(prompt, "").strip()
            sent_message = await event.reply(f"متن پیام صوتی:\n{cleaned_text}")
            
            # Start a new conversation thread for this transcription
            # This allows the user to continue the conversation with the transcribed text
            thread_id = event.id
            chat_session = gemini_model.start_chat()
            
            # Add initial context about the transcription
            chat_session.send_message(f"I transcribed a voice message from the user: {cleaned_text}")
            
            # Save conversation
            key = (event.chat_id, thread_id)
            conversations[key] = {
                "session": chat_session,
                "bot_message_ids": [sent_message.id],
                "original_user": event.sender_id
            }
        else:
            await event.reply("❌ متن پیام صوتی شناسایی نشد.")

    except Exception as e:
        await event.reply(f"خطا در تبدیل صوت به متن: {str(e)}")

    finally:
        if voice_path and os.path.exists(voice_path):
            os.remove(voice_path)

async def image_to_sticker_handler(event):
    """Convert images to stickers when replying with 'image to sticker' or 'i2s'"""
    try:
        # Check if it's a reply
        if not event.is_reply:
            return

        # Check if the command matches
        text = event.text.lower().strip()
        if text not in ["image to sticker", "i2s", "تبدیل به استیکر", "استیکر کن"]:
            return

        # Get the replied message
        reply_msg = await event.get_reply_message()
        
        # Check if the replied message has a photo
        if not reply_msg.photo:
            await event.reply("❌ لطفا به یک تصویر پاسخ دهید!")
            return

        # Status message
        status_msg = await event.reply("🔄 در حال ساخت استیکر...")
        
        # Download the photo
        photo_bytes = await reply_msg.download_media(file=io.BytesIO())
        
        # Convert to PIL Image
        image = Image.open(photo_bytes)
        
        # Resize image to sticker size (512x512 is recommended)
        if image.height >= image.width:
            new_height = 512
            new_width = int(new_height * image.width / image.height)
        else:
            new_width = 512
            new_height = int(new_width * image.height / image.width)
            
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create new image with transparent background
        background = Image.new('RGBA', (512, 512), (255, 255, 255, 0))
        offset = ((512 - new_width) // 2, (512 - new_height) // 2)
        background.paste(image, offset)
        
        # Save as WEBP
        output = io.BytesIO()
        background.save(output, format="WEBP")
        output.seek(0)
        output.name = "sticker.webp"
        
        # Send as a simple sticker
        await event.client.send_file(
            event.chat_id,
            output,
            force_document=False,
            reply_to=event.message.id
        )
        
        await status_msg.edit("✅ استیکر با موفقیت ساخته شد!")
        
    except Exception as e:
        await event.reply(f"❌ خطا در ساخت استیکر: {str(e)}")

async def fetch_page_content(url: str) -> str:
    """Get readable content from a webpage"""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer']):
            element.decompose()
            
        return ' '.join(soup.stripped_strings)[:3000]  # Limit to 3000 chars
    except Exception as e:
        return f"Failed to fetch content: {str(e)}"

async def search_web(query: str, num_results: int = 3) -> dict:
    """Perform Google search and return raw results"""
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://www.googleapis.com/customsearch/v1?q={encoded_query}&key={API_KEY}&cx={SEARCH_ENGINE_ID}&num={num_results}"
    
    try:
        response = requests.get(url)
        return response.json()
    except Exception as e:
        return {"error": f"Search error: {str(e)}"}

async def search_handler(event):
    """Handle web search requests"""
    try:
        # Extract query
        query = event.text.split("سرچ کن", 1)[1].strip()
        if not query:
            await event.reply("لطفا بعد از دستور سوال خود را بنویسید")
            return

        # Start processing
        processing_msg = await event.reply("🔍 در حال جستجو...")

        # Get search results
        search_results = await search_web(query)
        
        # Check for errors
        if "error" in search_results:
            await processing_msg.delete()
            await event.reply(f"❌ خطا در جستجو: {search_results['error']}")
            return
            
        if "items" not in search_results:
            await processing_msg.delete()
            await event.reply("❌ نتیجه‌ای یافت نشد")
            return

        # Get actual page contents
        contents = []
        for item in search_results['items'][:3]:  # Top 3 results
            content = await fetch_page_content(item['link'])
            contents.append(f"منبع {item['link']}:\n{content}")
        
        # Generate answer
        prompt = f"""
        با استفاده از این اطلاعات به زبان فارسی پاسخ بده:
        {''.join(contents)}
        
        سوال: {query}
        پاسخ:
        """
        
        # Start a chat session to maintain context
        chat_session = gemini_model.start_chat()
        response = chat_session.send_message(prompt)
        
        # Format response with sources
        sources = "\n".join([f"🔗 {item['link']}" for item in search_results['items'][:3]])
        final_response = f"{response.text}\n\nمنابع:\n{sources}"
        
        await processing_msg.delete()
        sent_message = await event.reply(final_response)
        
        # Start a new conversation thread for this search
        # This allows the user to reply to the search results
        thread_id = event.id
        
        # Save initial context about the search
        chat_session.send_message(f"User searched for '{query}' and I provided search results. My response included information from: {sources}")
        
        # Save conversation
        key = (event.chat_id, thread_id)
        conversations[key] = {
            "session": chat_session,
            "bot_message_ids": [sent_message.id],
            "original_user": event.sender_id
        }

    except Exception as e:
        await event.reply(f"خطا در جستجو: {str(e)}")

async def image_analysis_handler(event):
    """Handle direct image analysis requests"""
    try:
        # Constant for message length (defined at function level to avoid scope issues)
        MAX_MESSAGE_LENGTH = 4000  # Slightly less than 4096 for safety
        
        # Skip messages that are replies to our own messages (handled by reply_handler)
        if event.is_reply:
            reply_msg = await event.get_reply_message()
            
            # Get bot user info
            bot_info = await event.client.get_me()
            bot_id = bot_info.id
            
            # If replying to our bot with analysis keywords, let this handler handle it
            if reply_msg.sender_id == bot_id:
                is_analysis_request = False
                if event.text:
                    text_lower = event.text.lower()
                    specific_keywords = ["analyze", "analysis", "it", "this", "تحلیل", "آنالیز", "بررسی", "این", "تصویر"]
                    is_analysis_request = any(keyword in text_lower for keyword in specific_keywords)
                
                # If not specifically asking for analysis, let reply_handler handle it
                if not is_analysis_request:
                    return
            
            # If replying to a message without photo, skip
            if not reply_msg.photo and not event.photo:
                return
        
        # Check if this is a message with an image and text
        has_image = False
        if event.media and event.photo:
            has_image = True
        elif event.is_reply and reply_msg.photo:
            has_image = True
        
        if not has_image:
            return
        
        # Check if the message contains analysis request keywords
        has_analysis_keywords = False
        if event.text:
            text_lower = event.text.lower()
            keywords = ["analyze", "analysis", "it", "this", "تحلیل", "آنالیز", "بررسی", "چیست", "توضیح بده", "نظر بده", "چی هست", "تشخیص بده", "کن", "این", "تصویر"]
            has_analysis_keywords = any(keyword in text_lower for keyword in keywords)
        
        # If there's no text or no analysis keywords, don't process
        if not event.text or not has_analysis_keywords:
            return
            
        # Show processing message
        processing_msg = await event.reply("🔄 در حال تحلیل تصویر...")
        
        # Set a flag in the event object to mark it as being processed
        # This helps prevent double processing if multiple handlers match
        if hasattr(event, "_image_being_analyzed") and event._image_being_analyzed:
            await processing_msg.delete()
            return
        
        # Mark this event as being processed
        setattr(event, "_image_being_analyzed", True)
            
        # Get the image to analyze (either from the message or the replied message)
        try:
            # First try to get image from the message itself
            if event.photo:
                media_path = await event.download_media(file="temp_image.jpg")
            # If not available, try to get it from the replied message
            else:
                reply_msg = await event.get_reply_message()
                media_path = await reply_msg.download_media(file="temp_image.jpg")
            
            if not os.path.exists(media_path) or os.path.getsize(media_path) == 0:
                await processing_msg.delete()
                await event.reply("❌ خطا در دریافت تصویر. لطفا دوباره تلاش کنید.")
                return
        except Exception as download_error:
            await processing_msg.delete()
            await event.reply(f"❌ خطا در دریافت تصویر: {str(download_error)}")
            return
        
        try:
            # Analyze the image with system instructions
            result = await analyze_image(media_path, use_system_instructions=True)
            
            # Check if result is valid
            if not result or result.startswith(("Error", "\\x", "خطا")):
                if "payload too big" in result:
                    await event.reply("❌ تصویر بیش از حد بزرگ است. لطفا تصویر کوچکتری ارسال کنید.")
                else:
                    await event.reply(f"❌ {result}")
                return
                
            # Split long messages (Telegram has a 4096 character limit)
            if len(result) <= MAX_MESSAGE_LENGTH:
                # If the message is short enough, send it normally
                sent_message = await event.reply(result)
            else:
                # Split the message into chunks
                chunks = [result[i:i+MAX_MESSAGE_LENGTH] for i in range(0, len(result), MAX_MESSAGE_LENGTH)]
                
                # Send each chunk as a separate message
                first_message = None
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        # First chunk
                        first_message = await event.reply(chunk)
                    else:
                        # Subsequent chunks
                        await event.reply(f"(ادامه تحلیل - بخش {i+1}/{len(chunks)})\n\n{chunk}")
            
            # Start a new conversation thread for this analysis
            # This allows the user to reply to the analysis
            thread_id = event.id
            chat_session = gemini_model.start_chat()
            
            # Add initial context about the image
            chat_session.send_message(f"User asked about an image and I analyzed it with this response: {result}")
            
            # Save conversation
            key = (event.chat_id, thread_id)
            conversations[key] = {
                "session": chat_session,
                "bot_message_ids": [sent_message.id if 'sent_message' in locals() else first_message.id],
                "original_user": event.sender_id
            }
            
        finally:
            # Clean up
            if os.path.exists(media_path):
                os.remove(media_path)
            await processing_msg.delete()
            
    except Exception as e:
        await event.reply(f"⛔ خطا در تحلیل تصویر: {str(e)}")

async def image_generation_handler(event):
    """Handle image generation requests"""
    try:
        # Define output_file at the beginning with a default value
        output_file = ""
        processing_msg = None
        
        def save_binary(file_name, data):
            # Get directory path
            dir_path = os.path.dirname(file_name)
            
            # Create directory if it exists and isn't empty
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            # Write the file
            with open(file_name, 'wb') as f:
                f.write(data)
                
        def generate_image(prompt, output_file):
            client = genai2.Client(api_key="AIzaSyCHnUFbwJER66p1d3KPmaIUljDhsV1oHts")

            contents = [
                types2.Content(
                    role="user",
                    parts=[types2.Part.from_text(text=prompt)],
                )
            ]

            config = types2.GenerateContentConfig(
                temperature=1,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                response_modalities=["image", "text"],
                response_mime_type="text/plain",
            )

            try:
                response = client.models.generate_content_stream(
                    model="gemini-2.0-flash-exp-image-generation",
                    contents=contents,
                    config=config,
                )

                for chunk in response:
                    if chunk.candidates and chunk.candidates[0].content.parts:
                        part = chunk.candidates[0].content.parts[0]
                        if part.inline_data:
                            save_binary(output_file, part.inline_data.data)
                            print(f"Image saved to: {output_file}")
                        elif part.text:
                            print(part.text)
                            
            except Exception as e:
                print(f"Error generating image: {str(e)}")
                raise
            
        # Extract the prompt from the message
        prompt = event.text.split("عکس بساز", 1)[1].strip()
        
        if not prompt:
            await event.reply("لطفا بعد از عکس بساز دستورالعمل خود را بنویسید")
            return
        
        # Show processing message
        processing_msg = await event.reply("🎨 در حال ایجاد تصویر...")
        
        # Generate the image - use full absolute path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(current_dir, "temp_generated_image.png")
        
        try:
            # Make sure output_file is a valid path
            if not output_file:
                output_file = os.path.join(current_dir, "temp_generated_image.png")
                
            generate_image(prompt, output_file)
            
            # Check if file was actually created
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                await event.reply("⛔ خطا در ایجاد تصویر: فایل تصویر ایجاد نشد")
                if processing_msg:
                    await processing_msg.delete()
                return
                
            # Send the generated image
            await event.client.send_file(
                event.chat_id,
                output_file,
                force_document=False,  # Display as an image
                reply_to=event.message.id
            )
            
            if processing_msg:
                await processing_msg.delete()

        except Exception as e:
            await event.reply(f"⛔ خطا در ایجاد تصویر: {str(e)}")
            if processing_msg:
                await processing_msg.delete()
            
    except Exception as e:
        await event.reply(f"⛔ خطا در ایجاد تصویر: {str(e)}")
        if 'processing_msg' in locals() and processing_msg:
            await processing_msg.delete()

    finally:
        # Clean up temporary file
        if output_file and os.path.exists(output_file):
            try:
                os.remove(output_file)
            except Exception as e:
                pass

