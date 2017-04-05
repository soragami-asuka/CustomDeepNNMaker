// LayerDLLManager.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"
#include "Layer/NeuralNetwork/NNlayerFunction.h"
#include "LayerDLLManager.h"

#include<string>
#include<vector>

#include<boost/uuid/uuid_generators.hpp>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** DLLクラス */
	class NNLayerDLL : public ILayerDLL
	{
	protected:
		HMODULE hModule;

		FuncGetLayerCode funcGetLayerCode;
		FuncGetVersionCode funcGetVersionCode;
		
		FuncCreateLayerStructureSetting				funcCreateLayerStructureSetting;
		FuncCreateLayerStructureSettingFromBuffer	funcCreateLayerStructureSettingFromBuffer;

		FuncCreateLayerLearningSetting				funcCreateLearningSetting;
		FuncCreateLayerLearningSettingFromBuffer	funcCreateLearningSettingFromBuffer;

		FuncCreateLayerCPU funcCreateLayer;

		const ILayerDLLManager& layerDLLManager;

	public:
		/** コンストラクタ */
		NNLayerDLL(const ILayerDLLManager& i_layerDLLManager)
			:	hModule	(NULL)
			,	funcGetLayerCode							(NULL)
			,	funcGetVersionCode							(NULL)
			,	funcCreateLayerStructureSetting				(NULL)
			,	funcCreateLayerStructureSettingFromBuffer	(NULL)
			,	funcCreateLearningSetting					(NULL)
			,	funcCreateLearningSettingFromBuffer			(NULL)
			,	funcCreateLayer								(NULL)
			,	layerDLLManager								(i_layerDLLManager)
		{
		}
		/** デストラクタ */
		~NNLayerDLL()
		{
			if(this->hModule != NULL)
			{
				FreeLibrary(this->hModule);
				this->hModule = NULL;
			}
		}

	public:
		/** レイヤー識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		ErrorCode GetLayerCode(Gravisbell::GUID& o_layerCode)const
		{
			if(this->funcGetLayerCode == NULL)
				return ERROR_CODE_DLL_LOAD_FUNCTION;

			return this->funcGetLayerCode(o_layerCode);
		}
		/** バージョンコードを取得する.
			@param o_versionCode	格納先バッファ
			@return 成功した場合0 */
		ErrorCode GetVersionCode(VersionCode& o_versionCode)const
		{
			if(this->funcGetVersionCode == NULL)
				return ERROR_CODE_DLL_LOAD_FUNCTION;

			return this->funcGetVersionCode(o_versionCode);
		}


		/** レイヤー構造設定を作成する */
		SettingData::Standard::IData* CreateLayerStructureSetting(void)const
		{
			if(this->funcCreateLayerStructureSetting == NULL)
				return NULL;

			return this->funcCreateLayerStructureSetting();
		}
		/** レイヤー構造設定を作成する
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@param o_useBufferSize 実際に読み込んだバッファサイズ
			@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
		SettingData::Standard::IData* CreateLayerStructureSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)const
		{
			if(this->funcCreateLayerStructureSettingFromBuffer == NULL)
				return NULL;

			return this->funcCreateLayerStructureSettingFromBuffer(i_lpBuffer, i_bufferSize, o_useBufferSize);
		}


		/** レイヤー学習設定を作成する */
		SettingData::Standard::IData* CreateLearningSetting(void)const
		{
			if(this->funcCreateLearningSetting == NULL)
				return NULL;

			return this->funcCreateLearningSetting();
		}
		/** レイヤー学習設定を作成する
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@param o_useBufferSize 実際に読み込んだバッファサイズ
			@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
		SettingData::Standard::IData* CreateLearningSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)const
		{
			if(this->funcCreateLearningSettingFromBuffer == NULL)
				return NULL;

			return this->funcCreateLearningSettingFromBuffer(i_lpBuffer, i_bufferSize, o_useBufferSize);
		}

		
		/** CPU処理用のレイヤーを作成.
			GUIDは自動割り当て. */
		INNLayer* CreateLayer()const
		{
			boost::uuids::uuid uuid = boost::uuids::random_generator()();

			return this->CreateLayer(uuid.data);
		}
		/** CPU処理用のレイヤーを作成
			@param guid	作成レイヤーのGUID */
		INNLayer* CreateLayer(Gravisbell::GUID guid)const
		{
			if(this->funcCreateLayer == NULL)
				return NULL;

			return this->funcCreateLayer(guid, &this->layerDLLManager);
		}

	private:
		/** DLLをファイルから作成する(共通部分抜き出し) */
		static NNLayerDLL* CreateFromFileCommon(const ::std::wstring& filePath, const ILayerDLLManager& i_layerDLLManager)
		{
			// バッファを作成
			NNLayerDLL* pLayerDLL = new NNLayerDLL(i_layerDLLManager);
			if(pLayerDLL == NULL)
				return NULL;

			do
			{
				// DLLの読み込み
				pLayerDLL->hModule = LoadLibrary(filePath.c_str());
				if(pLayerDLL->hModule == NULL)
					break;

				// 関数読み込み
				// レイヤーコード
				pLayerDLL->funcGetLayerCode = (FuncGetLayerCode)GetProcAddress(pLayerDLL->hModule, "GetLayerCode");
				if(pLayerDLL->funcGetLayerCode == NULL)
					break;
				// バージョンコード
				pLayerDLL->funcGetVersionCode = (FuncGetVersionCode)GetProcAddress(pLayerDLL->hModule, "GetVersionCode");
				if(pLayerDLL->funcGetVersionCode == NULL)
					break;

				// レイヤー構造
				pLayerDLL->funcCreateLayerStructureSetting = (FuncCreateLayerStructureSetting)GetProcAddress(pLayerDLL->hModule, "CreateLayerStructureSetting");
				if(pLayerDLL->funcCreateLayerStructureSetting == NULL)
					break;
				pLayerDLL->funcCreateLayerStructureSettingFromBuffer = (FuncCreateLayerStructureSettingFromBuffer)GetProcAddress(pLayerDLL->hModule, "CreateLayerStructureSettingFromBuffer");
				if(pLayerDLL->funcCreateLayerStructureSettingFromBuffer == NULL)
					break;

				// 学習設定
				pLayerDLL->funcCreateLearningSetting = (FuncCreateLayerStructureSetting)GetProcAddress(pLayerDLL->hModule, "CreateLearningSetting");
				if(pLayerDLL->funcCreateLearningSetting == NULL)
					break;
				pLayerDLL->funcCreateLearningSettingFromBuffer = (FuncCreateLayerStructureSettingFromBuffer)GetProcAddress(pLayerDLL->hModule, "CreateLearningSettingFromBuffer");
				if(pLayerDLL->funcCreateLearningSettingFromBuffer == NULL)
					break;

				return pLayerDLL;
			}
			while(0);


			// DLLの作成に失敗.バッファを削除
			delete pLayerDLL;

			return NULL;
		}

	public:
		/** DLLをファイルから作成する */
		static NNLayerDLL* CreateFromFileCPU(const ::std::wstring& filePath, const ILayerDLLManager& i_layerDLLManager)
		{
			// 共通部分を作成
			NNLayerDLL* pLayerDLL = CreateFromFileCommon(filePath, i_layerDLLManager);
			if(pLayerDLL == NULL)
				return NULL;

			do
			{
				// レイヤー作成
				pLayerDLL->funcCreateLayer= (FuncCreateLayerCPU)GetProcAddress(pLayerDLL->hModule, "CreateLayerCPU");
				if(pLayerDLL->funcCreateLayer == NULL)
					break;

				return pLayerDLL;
			}
			while(0);


			// DLLの作成に失敗.バッファを削除
			delete pLayerDLL;

			return NULL;
		}
		/** DLLをファイルから作成する */
		static NNLayerDLL* CreateFromFileGPU(const ::std::wstring& filePath, const ILayerDLLManager& i_layerDLLManager)
		{
			// 共通部分を作成
			NNLayerDLL* pLayerDLL = CreateFromFileCommon(filePath, i_layerDLLManager);
			if(pLayerDLL == NULL)
				return NULL;

			do
			{
				// レイヤー作成
				pLayerDLL->funcCreateLayer= (FuncCreateLayerGPU)GetProcAddress(pLayerDLL->hModule, "CreateLayerGPU");
				if(pLayerDLL->funcCreateLayer == NULL)
					break;

				return pLayerDLL;
			}
			while(0);


			// DLLの作成に失敗.バッファを削除
			delete pLayerDLL;

			return NULL;
		}
	};


	/** DLL管理クラス */
	class LayerDLLManagerBase : public ILayerDLLManager
	{
	protected:
		std::vector<NNLayerDLL*> lppNNLayerDLL;

	public:
		/** コンストラクタ */
		LayerDLLManagerBase()
		{
		}
		/** デストラクタ */
		virtual ~LayerDLLManagerBase()
		{
			for(auto it : this->lppNNLayerDLL)
			{
				if(it != NULL)
					delete it;
			}
		}

	public:
		/** DLLを読み込んで、管理に追加する.
			@param szFilePath	読み込むファイルのパス.
			@param o_addLayerCode	追加されたGUIDの格納先アドレス.
			@return	成功した場合0が返る. */
		virtual ErrorCode ReadLayerDLL(const wchar_t szFilePath[], Gravisbell::GUID& o_addLayerCode) = 0;
		/** DLLを読み込んで、管理に追加する.
			@param szFilePath	読み込むファイルのパス.
			@return	成功した場合0が返る. */
		ErrorCode ReadLayerDLL(const wchar_t szFilePath[])
		{
			Gravisbell::GUID layerCode;
			return this->ReadLayerDLL(szFilePath, layerCode);
		}

		/** 管理しているレイヤーDLLの数を取得する */
		unsigned int GetLayerDLLCount()const
		{
			return this->lppNNLayerDLL.size();
		}
		/** 管理しているレイヤーDLLを番号指定で取得する.
			@param	num	取得するDLLの管理番号.
			@return 成功した場合はDLLクラスのアドレス. 失敗した場合はNULL */
		const ILayerDLL* GetLayerDLLByNum(unsigned int num)const
		{
			if(num >= this->lppNNLayerDLL.size())
				return NULL;

			return this->lppNNLayerDLL[num];
		}
		/** 管理しているレイヤーDLLをguid指定で取得する.
			@param guid	取得するDLLのGUID.
			@return 成功した場合はDLLクラスのアドレス. 失敗した場合はNULL */
		const ILayerDLL* GetLayerDLLByGUID(Gravisbell::GUID i_layerCode)const
		{
			for(unsigned int i=0; i<this->lppNNLayerDLL.size(); i++)
			{
				auto pLayerDLL = this->lppNNLayerDLL[i];
				if(pLayerDLL == NULL)
					continue;

				// GUIDを取得する
				Gravisbell::GUID layerCode;
				if(pLayerDLL->GetLayerCode(layerCode) != 0)
					continue;

				// 確認
				if(layerCode == i_layerCode)
					return pLayerDLL;
			}

			return NULL;
		}

		/** レイヤーDLLを削除する. */
		ErrorCode EraseLayerDLL(Gravisbell::GUID i_layerCode)
		{
			auto it = this->lppNNLayerDLL.begin();
			while(it != this->lppNNLayerDLL.end())
			{
				if(*it == NULL)
				{
					it++;
					continue;
				}

				// GUIDを取得する
				Gravisbell::GUID layerCode;
				if((*it)->GetLayerCode(layerCode) != 0)
				{
					it++;
					continue;
				}

				// 確認
				if(layerCode != i_layerCode)
				{
					it++;
					continue;
				}

				// 削除
				delete *it;
				this->lppNNLayerDLL.erase(it);
				return ERROR_CODE_NONE;
			}
			return ERROR_CODE_DLL_ERASE_NOTFOUND;
		}
	};

	/** DLL管理クラス(CPU処理) */
	class LayerDLLManagerCPU : public LayerDLLManagerBase
	{
	public:
		/** コンストラクタ */
		LayerDLLManagerCPU()
		{
		}
		/** デストラクタ */
		virtual ~LayerDLLManagerCPU()
		{
		}

	public:
		/** DLLを読み込んで、管理に追加する.
			@param szFilePath	読み込むファイルのパス.
			@param o_addLayerCode	追加されたGUIDの格納先アドレス.
			@return	成功した場合0が返る. */
		ErrorCode ReadLayerDLL(const wchar_t szFilePath[], Gravisbell::GUID& o_addLayerCode)
		{
			auto pLayerDLL = NNLayerDLL::CreateFromFileCPU(szFilePath, *this);
			if(pLayerDLL == NULL)
				return ERROR_CODE_DLL_LOAD_FUNCTION;

			Gravisbell::GUID guid;
			pLayerDLL->GetLayerCode(guid);

			// 管理レイヤーDLLから検索
			auto pLayerDLLAlready = this->GetLayerDLLByGUID(guid);
			if(pLayerDLLAlready != NULL)
			{
				// 既に追加済み
				delete pLayerDLL;
				return ERROR_CODE_DLL_ADD_ALREADY_SAMEID;
			}

			// 管理に追加
			this->lppNNLayerDLL.push_back(pLayerDLL);

			o_addLayerCode = guid;

			return ERROR_CODE_NONE;
		}
	};
	/** DLL管理クラス(GPU処理) */
	class LayerDLLManagerGPU : public LayerDLLManagerBase
	{
	public:
		/** コンストラクタ */
		LayerDLLManagerGPU()
		{
		}
		/** デストラクタ */
		virtual ~LayerDLLManagerGPU()
		{
		}

	public:
		/** DLLを読み込んで、管理に追加する.
			@param szFilePath	読み込むファイルのパス.
			@param o_addLayerCode	追加されたGUIDの格納先アドレス.
			@return	成功した場合0が返る. */
		ErrorCode ReadLayerDLL(const wchar_t szFilePath[], Gravisbell::GUID& o_addLayerCode)
		{
			auto pLayerDLL = NNLayerDLL::CreateFromFileGPU(szFilePath, *this);
			if(pLayerDLL == NULL)
				return ERROR_CODE_DLL_LOAD_FUNCTION;

			Gravisbell::GUID guid;
			pLayerDLL->GetLayerCode(guid);

			// 管理レイヤーDLLから検索
			auto pLayerDLLAlready = this->GetLayerDLLByGUID(guid);
			if(pLayerDLLAlready != NULL)
			{
				// 既に追加済み
				delete pLayerDLL;
				return ERROR_CODE_DLL_ADD_ALREADY_SAMEID;
			}

			// 管理に追加
			this->lppNNLayerDLL.push_back(pLayerDLL);

			o_addLayerCode = guid;

			return ERROR_CODE_NONE;
		}
	};


	// DLL管理クラスを作成
	extern LayerDLLManager_API ILayerDLLManager* CreateLayerDLLManagerCPU()
	{
		return new LayerDLLManagerCPU();
	}
	// DLL管理クラスを作成
	extern LayerDLLManager_API ILayerDLLManager* CreateLayerDLLManagerGPU()
	{
		return new LayerDLLManagerGPU();
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell

