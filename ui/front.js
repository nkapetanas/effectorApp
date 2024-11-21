import React, { useState } from 'react';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Settings } from 'lucide-react';
import { Label } from '@/components/ui/label';

const EffectorUI = () => {
    const [selectedMethod, setSelectedMethod] = useState('pdp');
    const [featureIndex, setFeatureIndex] = useState(0);
    const [plotImage, setPlotImage] = useState(null);
    const [loading, setLoading] = useState(false);
    const [config, setConfig] = useState({
        dataUrl: '',
        modelUrl: '',
        publishUrl: ''
    });

    const handleAnalyze = async () => {
        try {
            setLoading(true);

            const response = await fetch('http://localhost:5000/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    method: selectedMethod,
                    feature_index: featureIndex,
                    data_url: config.dataUrl,
                    model_url: config.modelUrl,
                    publish_url: config.publishUrl || null
                }),
            });

            const data = await response.json();

            if (data.status === 'success') {
                setPlotImage(`data:image/png;base64,${data.results[`${selectedMethod}_plot`]}`);
                if (!data.published && config.publishUrl) {
                    console.warn('Failed to publish results to specified URL');
                }
            }
        } catch (error) {
            console.error('Error:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="p-6 max-w-4xl mx-auto">
            <Card>
                <CardHeader>
                    <h2 className="text-2xl font-bold">Effector Analysis Dashboard</h2>
                </CardHeader>
                <CardContent>
                    <div className="space-y-4">
                        <div className="grid grid-cols-1 gap-4">
                            <div>
                                <Label className="text-sm font-medium mb-1">Data URL</Label>
                                <Input
                                    value={config.dataUrl}
                                    onChange={(e) => setConfig(prev => ({...prev, dataUrl: e.target.value}))}
                                    placeholder="https://api.example.com/data"
                                />
                            </div>
                            <div>
                                <Label className="text-sm font-medium mb-1">Model URL</Label>
                                <Input
                                    value={config.modelUrl}
                                    onChange={(e) => setConfig(prev => ({...prev, modelUrl: e.target.value}))}
                                    placeholder="https://api.example.com/model"
                                />
                            </div>
                            <div>
                                <Label className="text-sm font-medium mb-1">Results Publish URL (Optional)</Label>
                                <Input
                                    value={config.publishUrl}
                                    onChange={(e) => setConfig(prev => ({...prev, publishUrl: e.target.value}))}
                                    placeholder="https://api.example.com/publish"
                                />
                            </div>
                        </div>

                        <div className="flex gap-4 items-center">
                            <div>
                                <Label className="text-sm font-medium mb-1">Analysis Method</Label>
                                <Select value={selectedMethod} onValueChange={setSelectedMethod}>
                                    <SelectTrigger className="w-32">
                                        <SelectValue placeholder="Select method" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="pdp">PDP</SelectItem>
                                        <SelectItem value="rhale">RHALE</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>

                            <div>
                                <Label className="text-sm font-medium mb-1">Feature Index to Analyze</Label>
                                <div className="flex items-center gap-2">
                                    <Input
                                        type="number"
                                        value={featureIndex}
                                        onChange={(e) => setFeatureIndex(parseInt(e.target.value))}
                                        className="w-20"
                                        min="0"
                                        title="Select which feature to analyze (0-based index)"
                                    />
                                    <span className="text-sm text-gray-500">
                    (0-based index of feature to analyze)
                  </span>
                                </div>
                            </div>

                            <div className="flex flex-col justify-end">
                                <Button
                                    onClick={handleAnalyze}
                                    className="flex items-center gap-2"
                                    disabled={loading || !config.dataUrl || !config.modelUrl}
                                >
                                    {loading ? <Settings className="w-4 h-4 animate-spin" /> : <Settings className="w-4 h-4" />}
                                    {loading ? 'Analyzing...' : 'Analyze'}
                                </Button>
                            </div>
                        </div>

                        {plotImage && (
                            <div className="mt-4">
                                <img src={plotImage} alt="Analysis Plot" className="w-full" />
                            </div>
                        )}
                    </div>
                </CardContent>
            </Card>
        </div>
    );
};

export default EffectorUI;