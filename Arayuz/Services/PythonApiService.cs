// PythonApiService.cs
using System.Text.Json;
using OtobusDashboard.Models;
namespace OtobusDashboard.Services
{
    public class PythonApiService
    {
        private readonly HttpClient _http;
        private readonly string _baseUrl;
        private readonly JsonSerializerOptions _jsonOptions;
        public PythonApiService(HttpClient http, IConfiguration config)
        {
            _http = http;
            _baseUrl = config["PythonApi:BaseUrl"] ?? "http://localhost:5000";
            _jsonOptions = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };
        }
        // --- YOLCU API ÇAĞRILARI (Mevcut) ---
        public async Task<KaynakResponse?> GetKaynaklarAsync()
        {
            try
            {
                var json = await _http.GetStringAsync($"{_baseUrl}/kaynaklar");
                return JsonSerializer.Deserialize<KaynakResponse>(json, _jsonOptions);
            }
            catch { return null; }
        }
        public async Task<bool> KaynakSecAsync(int kaynakId)
        {
            try
            {
                var res = await _http.PostAsync($"{_baseUrl}/kaynaklar/{kaynakId}/sec", null);
                return res.IsSuccessStatusCode;
            }
            catch { return false; }
        }
        public async Task<AnlikVeri?> GetAnlikVeriAsync()
        {
            try
            {
                var json = await _http.GetStringAsync($"{_baseUrl}/canli");
                return JsonSerializer.Deserialize<AnlikVeri>(json, _jsonOptions);
            }
            catch { return null; }
        }
        public async Task<OturumlarResponse?> GetOturumlarAsync()
        {
            try
            {
                var json = await _http.GetStringAsync($"{_baseUrl}/db/oturumlar");
                return JsonSerializer.Deserialize<OturumlarResponse>(json, _jsonOptions);
            }
            catch { return null; }
        }
        public async Task<GecmisResponse?> GetGecmisAsync(int limit = 50)
        {
            try
            {
                var json = await _http.GetStringAsync($"{_baseUrl}/db/gecmis?limit={limit}");
                return JsonSerializer.Deserialize<GecmisResponse>(json, _jsonOptions);
            }
            catch { return null; }
        }
        // ==========================================================
        // --- YENİ EKLENEN SÜRÜCÜ API ÇAĞRILARI ---
        // ==========================================================
        public async Task<SurucuAnlikVeri?> GetSurucuAnlikVeriAsync()
        {
            try
            {
                var json = await _http.GetStringAsync($"{_baseUrl}/surucu/durum");
                return JsonSerializer.Deserialize<SurucuAnlikVeri>(json, _jsonOptions);
            }
            catch { return null; }
        }
        public async Task<SurucuIstatistik?> GetSurucuIstatistikAsync()
        {
            try
            {
                var json = await _http.GetStringAsync($"{_baseUrl}/surucu/istatistik");
                return JsonSerializer.Deserialize<SurucuIstatistik>(json, _jsonOptions);
            }
            catch { return null; }
        }
        public async Task<YanginAnlikVeri?> GetYanginAnlikVeriAsync()
        {
            try
            {
                var json = await _http.GetStringAsync($"{_baseUrl}/yangin/durum");
                return JsonSerializer.Deserialize<YanginAnlikVeri>(json, _jsonOptions);
            }
            catch { return null; }
        }
        public async Task<YanginGecmisResponse?> GetYanginGecmisAsync(int limit = 20)
        {
            try
            {
                var json = await _http.GetStringAsync($"{_baseUrl}/yangin/gecmis?limit={limit}");
                return JsonSerializer.Deserialize<YanginGecmisResponse>(json, _jsonOptions);
            }
            catch { return null; }
        }


    }
}
