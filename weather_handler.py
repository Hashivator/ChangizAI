from google import genai
from google.genai import types
import json
import asyncio  # Add this import
import re

async def weather_handler(event):
    """Handle weather requests"""
    try:
        # Synchronous Gemini call wrapped in async thread
        def _get_weather_sync(city):
            client = genai.Client(api_key="AIzaSyCHnUFbwJER66p1d3KPmaIUljDhsV1oHts")
            tools = [types.Tool(google_search=types.GoogleSearch())]
            
            generate_content_config = types.GenerateContentConfig(
                temperature=1,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                tools=tools,
                response_mime_type="text/plain",
                system_instruction=types.Part.from_text(text="""شما باید حتما و فقط JSON معتبر برگردانید.
ساختار مورد نیاز:
{
    "city": "نام شهر به فارسی",
    "temperature": "دما به سانتیگراد (مثال: ۲۵°C)",
    "humidity": "رطوبت (مثال: ۴۵٪)",
    "wind": "سرعت باد (مثال: ۱۰ کیلومتر بر ساعت)",
    "additional_info": "توضیحات اضافه",
    "feeling": "احساس آب و هوا به فارسی"
}
هرگز از ```json یا علامتهای اضافه استفاده نکنید!
فقط و فقط JSON برگردانده شود
هیچ جوابی جز JSON برگردانده نشود حتی اگه نتونی اطلاعات آب و هوای حال حاضر رو بدی و فقط براساس سرچ و تخمین جواب بده
تاکید میکنم فقط و فقط JSON برگردانده شود""")

            )

            contents = [types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"آب و هوای {city}")]
            )]

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=generate_content_config,
            )
            return response.text

        # Async wrapper for the sync call
        async def get_weather(city):
            return await asyncio.to_thread(_get_weather_sync, city)

        # Response analyzer remains the same
        def analyze_weather_response(response):
            try:
                # First approach: Try to directly parse the response as JSON
                try:
                    weather_data = json.loads(response)
                    print("Parsed directly as JSON")
                except json.JSONDecodeError:
                    # Second approach: Try to find JSON pattern using regex
                    pattern = r"```json\s*([\s\S]*?)\s*```"
                    match = re.search(pattern, response, re.DOTALL)
                    if match:
                        json_str = match.group(1).strip()
                        weather_data = json.loads(json_str)
                        print("Extracted JSON using regex with code blocks")
                    else:
                        # Third approach: Try a more robust pattern to find a complete JSON object
                        try:
                            # Find the first opening brace and the last closing brace
                            start_idx = response.find('{')
                            if start_idx != -1:
                                # Count braces to find matching end
                                brace_count = 0
                                for i, char in enumerate(response[start_idx:]):
                                    if char == '{':
                                        brace_count += 1
                                    elif char == '}':
                                        brace_count -= 1
                                        if brace_count == 0:
                                            # Found the complete JSON object
                                            json_str = response[start_idx:start_idx+i+1]
                                            weather_data = json.loads(json_str)
                                            print("Extracted JSON using brace counting")
                                            break
                                else:
                                    # If we didn't break out of the loop, we didn't find a complete object
                                    raise ValueError("No complete JSON object found")
                            else:
                                raise ValueError("No opening brace found")
                        except Exception as inner_e:
                            print(f"Error in brace counting approach: {inner_e}")
                            # Fall back to the simple pattern
                            pattern = r"{.*}"
                            match = re.search(pattern, response, re.DOTALL)
                            if match:
                                json_str = match.group(0).strip()
                                weather_data = json.loads(json_str)
                                print("Extracted JSON using regex for {} pattern")
                            else:
                                # Last resort: Try removing markdown code blocks
                                cleaned = response.replace("```json", "").replace("```", "").strip()
                                weather_data = json.loads(cleaned)
                                print("Extracted JSON by cleaning markdown")
                
                print("Weather data:", weather_data)
                return f"""🏙️ شهر: {weather_data.get('city', '?')}
🌡️ دما: {weather_data.get('temperature', '?')}
💧 رطوبت: {weather_data.get('humidity', '?')}
🌬️ باد: {weather_data.get('wind', '?')}
😊 احساس: {weather_data.get('feeling', '?')}

ℹ️ اطلاعات بیشتر: {weather_data.get('additional_info', '?')}"""
            except Exception as e:
                print(f"Error parsing JSON: {e}, Response: {response}")
                return f"⛔ خطا: پاسخ نامعتبر از سرور ({e})"

        # Extract city name
        message_text = event.text.strip()
        city = next((message_text.split(prefix, 1)[1].strip() 
                   for prefix in ["آب و هوای", "وضعیت هوای", "هوای", "آب و هوا", "وضعیت هوا"] 
                   if prefix in message_text), None)

        if not city:
            await event.reply("لطفا نام شهر را به درستی وارد کنید. مثال: هوای تهران")
            return

        # Await the Gemini API call
        weather_response = await get_weather(city)
        formatted_response = analyze_weather_response(weather_response)
        await event.reply(formatted_response)

    except Exception as e:
        await event.reply(f"⛔ خطا: {str(e)}")