import { useState } from 'react'
import './App.css'
import { Button } from './components/ui/button'
import { Textarea } from './components/ui/textarea'
import { Skeleton } from './components/ui/skeleton'

function App() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [prompt, setPrompt] = useState('');
  const fetchImage = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/generate?prompt=${prompt}`);
      if (!response.ok) {
        throw new Error('Image generation failed');
      }

      // Get the response as a Blob (binary data)
      const blob = await response.blob();

      // Create an Object URL for the image Blob to use it in an <img> tag
      const url = URL.createObjectURL(blob);

      // Set the image URL in the state
      setImageUrl(url);

    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  if (loading) {
    return (
      <div>
        <h1 className='mb-2'>PyTorch Stable Diffusion</h1>
        <div className='flex justify-center items-center w-full'>
          <Skeleton className="w-1/2 h-[250px]" />
        </div>
        <div className="card">
          <Textarea disabled placeholder="Type your prompt here." />
        </div>
        <Button disabled>Generate</Button>
        <p className="read-the-docs mt-4">
          Find the source code on Github
        </p>
      </div>
    )
  }

  return (
    <div>

      <h1 className='mb-2'>PyTorch Stable Diffusion</h1>
      {imageUrl &&
        <div className='flex justify-center items-center w-full'>
          <img
            className='w-1/2 items-center'
            src={imageUrl} alt="Generated image" />
        </div>
      }
      <div className="card">
        <Textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Type your prompt here." />
      </div>
      <Button onClick={fetchImage}>Generate</Button>
      <p className="read-the-docs mt-4">
        Find the source code on Github
      </p>
    </div>
  )
}

export default App
