//=========================================
// ニューラルネットワーク用レイヤーデータの管理クラス
//=========================================
#include"stdafx.h"

#include<map>

#include"Library/NeuralNetwork/LayerDataManager.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class LayerDataManager : public ILayerDataManager
	{
	private:
		std::map<Gravisbell::GUID, ILayerData*> lpLayerData;

	public:
		LayerDataManager()
			:	ILayerDataManager()
		{
		}
		virtual ~LayerDataManager()
		{
			this->ClearLayerData();
		}

	public:
		/** レイヤーデータの作成.内部的に管理まで行う.
			@param	i_layerDLLManager	レイヤーDLL管理クラス.
			@param	i_typeCode			レイヤー種別コード
			@param	i_guid				新規作成するレイヤーデータのGUID
			@param	i_layerStructure	レイヤー構造
			@param	i_inputDataStruct	入力データ構造
			@param	o_pErrorCode		エラーコード格納先のアドレス. NULL指定可.
			@return
			typeCodeが存在しない場合、NULLを返す.
			既に存在するguidでtypeCodeも一致した場合、内部保有のレイヤーデータを返す.
			既に存在するguidでtypeCodeが異なる場合、NULLを返す. */
		ILayerData* CreateLayerData(
			const ILayerDLLManager& i_layerDLLManager,
			const Gravisbell::GUID& i_typeCode,
			const Gravisbell::GUID& i_guid,
			const SettingData::Standard::IData& i_layerStructure,
			Gravisbell::ErrorCode* o_pErrorCode = NULL)
		{
			// 同一レイヤーが存在しないか確認
			if(this->lpLayerData.count(i_guid))
			{
				if(lpLayerData[i_guid]->GetLayerCode() == i_typeCode)
				{
					return lpLayerData[i_guid];
				}
				else
				{
					if(o_pErrorCode)
						*o_pErrorCode = ErrorCode::ERROR_CODE_ADDLAYER_ALREADY_SAMEID;
					return NULL;
				}
			}

			// DLLを検索
			auto pLayerDLL = i_layerDLLManager.GetLayerDLLByGUID(i_typeCode);
			if(pLayerDLL == NULL)
			{
				if(o_pErrorCode)
					*o_pErrorCode = ErrorCode::ERROR_CODE_DLL_NOTFOUND;
				return NULL;
			}

			// レイヤーデータを作成
			auto pLayerData = pLayerDLL->CreateLayerData(i_guid, i_layerStructure);
			if(pLayerData == NULL)
			{
				if(o_pErrorCode)
					*o_pErrorCode = ErrorCode::ERROR_CODE_LAYER_CREATE;
				return NULL;
			}

			// レイヤーデータを保存
			this->lpLayerData[i_guid] = pLayerData;

			if(o_pErrorCode)
				*o_pErrorCode = ErrorCode::ERROR_CODE_NONE;

			return pLayerData;
		}



		/** レイヤーデータをバッファから作成.内部的に管理まで行う.
			@param	i_layerDLLManager	レイヤーDLL管理クラス.
			@param	i_typeCode			レイヤー種別コード
			@param	i_guid				新規作成するレイヤーデータのGUID
			@param	i_lpBuffer		読み取り用バッファ.
			@param	i_bufferSize	使用可能なバッファサイズ.
			@param	o_useBufferSize	実際に使用したバッファサイズ.
			@param	o_pErrorCode		エラーコード格納先のアドレス. NULL指定可.
			@return
			typeCodeが存在しない場合、NULLを返す.
			既に存在するguidでtypeCodeも一致した場合、内部保有のレイヤーデータを返す.
			既に存在するguidでtypeCodeが異なる場合、NULLを返す. */
		ILayerData* CreateLayerData(
			const ILayerDLLManager& i_layerDLLManager,
			const Gravisbell::GUID& i_typeCode,
			const Gravisbell::GUID& i_guid,
			const BYTE* i_lpBuffer, S64 i_bufferSize, S64& o_useBufferSize,
			Gravisbell::ErrorCode* o_pErrorCode = NULL)
		{
			// DLLを検索
			auto pLayerDLL = i_layerDLLManager.GetLayerDLLByGUID(i_typeCode);
			if(pLayerDLL == NULL)
			{
				if(o_pErrorCode)
					*o_pErrorCode = ErrorCode::ERROR_CODE_DLL_NOTFOUND;
				return NULL;
			}

			// レイヤーデータを作成
			auto pLayerData = pLayerDLL->CreateLayerDataFromBuffer(i_guid, i_lpBuffer, i_bufferSize, o_useBufferSize);
			if(pLayerData == NULL)
			{
				if(o_pErrorCode)
					*o_pErrorCode = ErrorCode::ERROR_CODE_LAYER_CREATE;
				return NULL;
			}


			// 同一レイヤーが存在しないか確認
			if(this->lpLayerData.count(i_guid))
			{
				delete pLayerData;
				if(lpLayerData[i_guid]->GetLayerCode() == i_typeCode)
				{
					return lpLayerData[i_guid];
				}
				else
				{
					if(o_pErrorCode)
						*o_pErrorCode = ErrorCode::ERROR_CODE_ADDLAYER_ALREADY_SAMEID;
					return NULL;
				}
			}

			// レイヤーデータを保存
			this->lpLayerData[i_guid] = pLayerData;

			if(o_pErrorCode)
				*o_pErrorCode = ErrorCode::ERROR_CODE_NONE;

			return pLayerData;
		}


		/** レイヤーデータをGUID指定で取得する */
		ILayerData* GetLayerData(const Gravisbell::GUID& i_guid)
		{
			if(this->lpLayerData.count(i_guid))
				return this->lpLayerData[i_guid];
			return NULL;
		}

		/** レイヤーデータ数を取得する */
		U32 GetLayerDataCount()
		{
			return (U32)this->lpLayerData.size();
		}
		/** レイヤーデータを番号指定で取得する */
		ILayerData* GetLayerDataByNum(U32 i_num)
		{
			if(i_num >= this->lpLayerData.size())
				return NULL;

			auto it = this->lpLayerData.begin();
			for(U32 i=0; i<i_num; i++)
				it++;

			return it->second;
		}

		/** レイヤーデータをGUID指定で削除する */
		Gravisbell::ErrorCode EraseLayerByGUID(const Gravisbell::GUID& i_guid)
		{
			auto it = this->lpLayerData.find(i_guid);
			if(it == this->lpLayerData.end())
				return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;

			delete it->second;
			this->lpLayerData.erase(it);

			return ErrorCode::ERROR_CODE_NONE;
		}
		/** レイヤーデータを番号指定で削除する */
		Gravisbell::ErrorCode EraseLayerByNum(U32 i_num)
		{
			if(i_num >= this->lpLayerData.size())
				return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			auto it = this->lpLayerData.begin();
			for(U32 i=0; i<i_num; i++)
				it++;

			delete it->second;
			this->lpLayerData.erase(it);

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** レイヤーデータをすべて削除する */
		Gravisbell::ErrorCode ClearLayerData()
		{
			auto it = this->lpLayerData.begin();
			while(it != this->lpLayerData.end())
			{
				delete it->second;
				it = this->lpLayerData.erase(it);
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** レイヤーデータの管理クラスを作成 */
	LayerDataManager_API ILayerDataManager* CreateLayerDataManager()
	{
		return new LayerDataManager();
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell