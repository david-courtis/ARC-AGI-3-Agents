> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# API Keys

> How to get and use your ARC-AGI-3 API key

## Why Get an API Key?

Registering for an API key allows you to:

* **Track your progress** across games and sessions
* **Access the full list of games** when launch goes out

## How to Get Your API Key

1. Go to [arcprize.org/platform](https://arcprize.org/platform)

2. Register by logging in with either **Google** or **GitHub**

3. Click on your **user profile** in the top right corner

4. In your user profile, find the **API Keys** section

5. Create a new key. This is your `ARC_AGI_API` key

Once you have your key, set it in your enviornment or `.env` and you'll have access to the entire set of public games once available on the platform.

## Using Your API Key

Set your API key as an environment variable:

```bash  theme={null}
export ARC_API_KEY="your-api-key-here"
```

Or add it to a `.env` file in your project:

```bash  theme={null}
echo 'ARC_API_KEY=your-api-key-here' > .env
```

The toolkit will automatically load your key from the environment when you create an `Arcade` instance:

```python  theme={null}
import arc_agi

# Automatically uses ARC_API_KEY from environment
arc = arc_agi.Arcade()

# Or pass the API key explicitly
arc = arc_agi.Arcade(arc_api_key="your-api-key-here")
```


Built with [Mintlify](https://mintlify.com).