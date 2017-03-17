// Data.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"

#include"SettingData.h"

#include"Common/VersionCode.h"

#include<string>
#include<vector>

namespace Gravisbell {
namespace SettingData {
namespace Standard {

	class Data : public IDataEx
	{
	private:
		GUID layerCode;
		VersionCode versionCode;

		std::vector<IItemBase*> lpLayerConfigItem;

	public:
		/** コンストラクタ */
		Data(const GUID& layerCode, const VersionCode& versionCode)
			: IDataEx()
			, layerCode		(layerCode)
			, versionCode	(versionCode)
		{
		}
		/** コピーコンストラクタ */
		Data(const Data& config)
			: layerCode	(config.layerCode)
			, versionCode (config.versionCode)
		{
			for(unsigned int itemNum=0; itemNum<config.lpLayerConfigItem.size(); itemNum++)
			{
				this->lpLayerConfigItem.push_back(config.lpLayerConfigItem[itemNum]->Clone());
			}
		}
		/** デストラクタ */
		virtual ~Data()
		{
			for(unsigned int itemNum=0; itemNum<lpLayerConfigItem.size(); itemNum++)
			{
				if(lpLayerConfigItem[itemNum] != NULL)
					delete lpLayerConfigItem[itemNum];
			}
		}

		/** 一致演算 */
		bool operator==(const IData& config)const
		{
			Data* pConfig = (Data*)&config;

			// レイヤーコードの確認
			if(this->layerCode != pConfig->layerCode)
				return false;
			// バージョンコードの確認
			if(this->versionCode != pConfig->versionCode)
				return false;

			// アイテム数の確認
			if(this->lpLayerConfigItem.size() != pConfig->lpLayerConfigItem.size())
				return false;

			// 各アイテムの確認
			for(unsigned int itemNum=0; itemNum<this->lpLayerConfigItem.size(); itemNum++)
			{
				IItemBase* pItemL = this->lpLayerConfigItem[itemNum];
				IItemBase* pItemR = pConfig->lpLayerConfigItem[itemNum];

				// どちらか片方がNULLだった場合は終了
				if((pItemL == NULL) ^ (pItemR == NULL))
					return false;

				// 両方NULLだった場合は一致と判定
				if((pItemL == NULL) && (pItemR == NULL))
					continue;

				// 各アイテムを確認
				if(*pItemL != *pItemR)
					return false;
			}

			return true;
		}
		/** 不一致演算 */
		bool operator!=(const IData& config)const
		{
			return !(*this == config);
		}

		/** 自身の複製を作成する */
		virtual IData* Clone()const
		{
			return new Data(*this);
		}

	public:
		/** レイヤー識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		ErrorCode GetLayerCode(GUID& o_guid)const
		{
			o_guid = this->layerCode;

			return ERROR_CODE_NONE;
		}

	public:
		/** 設定項目数を取得する */
		unsigned int GetItemCount()const
		{
			return this->lpLayerConfigItem.size();
		}
		/** 設定項目を番号指定で取得する */
		IItemBase* GetItemByNum(U32 i_num)
		{
			return const_cast<IItemBase*>((static_cast<const Data&>(*this)).GetItemByNum(i_num));
		}
		/** 設定項目を番号指定で取得する */
		const IItemBase* GetItemByNum(U32 i_num)const
		{
			if(i_num >= this->lpLayerConfigItem.size())
				return NULL;
			return this->lpLayerConfigItem[i_num];
		}
		/** 設定項目をID指定で取得する */
		IItemBase* GetItemByID(const wchar_t i_szIDBuf[])
		{
			return const_cast<IItemBase*>((static_cast<const Data&>(*this)).GetItemByID(i_szIDBuf));
		}
		/** 設定項目をID指定で取得する */
		const IItemBase* GetItemByID(const wchar_t i_szIDBuf[])const
		{
			// 同一IDを検索
			for(unsigned int i=0; i<this->lpLayerConfigItem.size(); i++)
			{
				wchar_t szID[ITEM_NAME_MAX];

				// 対象項目のIDを取得
				if(this->lpLayerConfigItem[i]->GetConfigID(szID) != ErrorCode::ERROR_CODE_NONE)
					continue;

				// 比較
				if(std::wstring(szID) == i_szIDBuf)
					return this->lpLayerConfigItem[i];
			}
			return NULL;
		}
		
		/** アイテムを追加する.
			追加されたアイテムは内部でdeleteされる. */
		int AddItem(IItemBase* pItem)
		{
			if(pItem == NULL)
				return -1;

			this->lpLayerConfigItem.push_back(pItem);

			return 0;
		}

	public:
		/** 保存に必要なバイト数を取得する */
		unsigned int GetUseBufferByteCount()const
		{
			unsigned int bufferSize = 0;

			// レイヤーコード
			bufferSize += sizeof(this->layerCode);

			// バージョンコード
			bufferSize += sizeof(this->versionCode);

			// アイテム数
			bufferSize += sizeof(unsigned int);

			// 各アイテム
			for(unsigned int itemNum=0; itemNum<this->lpLayerConfigItem.size(); itemNum++)
			{
				// アイテム種別
				bufferSize += sizeof(ItemType);

				// アイテム
				bufferSize += this->lpLayerConfigItem[itemNum]->GetUseBufferByteCount();
			}

			return bufferSize;
		}

		/** バッファから読み込む.
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
		int ReadFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize)
		{
			if(i_bufferSize < (int)this->GetUseBufferByteCount())
				return -1;

			unsigned int bufferPos = 0;

			// レイヤーコード
			GUID tmpLayerCode = *(GUID*)&i_lpBuffer[bufferPos];
			bufferPos += sizeof(GUID);

			// チェック
			if(this->layerCode != tmpLayerCode)
				return -1;


			// バージョンコード
			VersionCode tmpVersionCode = *(VersionCode*)&i_lpBuffer[bufferPos];
			bufferPos += sizeof(this->versionCode);

			// チェック
			if(this->versionCode.major != tmpVersionCode.major)
				return -1;
			if(this->versionCode.minor != tmpVersionCode.minor)
				return -1;
			if(this->versionCode.revision != tmpVersionCode.revision)
				return -1;

			this->versionCode = tmpVersionCode;


			// アイテム数
			unsigned int itemCount = *(unsigned int*)&i_lpBuffer[bufferPos];
			bufferPos += sizeof(unsigned int);

			// チェック
			if(this->lpLayerConfigItem.size() != itemCount)
				return -1;



			std::vector<IItemBase*> lpTmpItem;
			for(unsigned int itemNum=0; itemNum<itemCount; itemNum++)
			{
				// アイテム種別
				ItemType itemType = *(ItemType*)&i_lpBuffer[bufferPos];
				bufferPos += sizeof(ItemType);

				if(this->lpLayerConfigItem[itemNum]->GetItemType() != itemType)
					return -1;

				// 読み込み
				bufferPos += this->lpLayerConfigItem[itemNum]->ReadFromBuffer(&i_lpBuffer[bufferPos], i_bufferSize - bufferPos);
			}


			return bufferPos;
		}
		/** バッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合0 */
		int WriteToBuffer(BYTE* o_lpBuffer)const
		{
			unsigned int bufferPos = 0;
			
			// レイヤーコード
			*(GUID*)&o_lpBuffer[bufferPos] = this->layerCode;
			bufferPos += sizeof(GUID);

			// バージョンコード
			*(VersionCode*)&o_lpBuffer[bufferPos] = this->versionCode;
			bufferPos += sizeof(VersionCode);

			// アイテム数
			*(unsigned int*)&o_lpBuffer[bufferPos] = this->lpLayerConfigItem.size();
			bufferPos += sizeof(unsigned int);

			// 各アイテム
			for(unsigned int itemNum=0; itemNum<this->lpLayerConfigItem.size(); itemNum++)
			{
				// アイテム種別
				*(ItemType*)&o_lpBuffer[bufferPos] = this->lpLayerConfigItem[itemNum]->GetItemType();
				bufferPos += sizeof(ItemType);

				// アイテム
				bufferPos += this->lpLayerConfigItem[itemNum]->WriteToBuffer(&o_lpBuffer[bufferPos]);
			}

			return bufferPos;
		}

	public:
		/** 構造体にデータを格納する.
			@param	o_lpBuffer	構造体の先頭アドレス. 構造体はConvertNNCofigToSourceから出力されたソースを使用する. */
		ErrorCode WriteToStruct(BYTE* o_lpBuffer)const
		{
			unsigned int bufferPos = 0;
			
			// 各アイテム
			for(unsigned int itemNum=0; itemNum<this->lpLayerConfigItem.size(); itemNum++)
			{
				const IItemBase* pItemBase = this->GetItemByNum(itemNum);
				if(pItemBase == NULL)
					continue;

				bufferPos += pItemBase->WriteToStruct(&o_lpBuffer[bufferPos]);
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
		/** 構造体からデータを読み込む.
			@param	o_lpBuffer	構造体の先頭アドレス. 構造体はConvertNNCofigToSourceから出力されたソースを使用する. */
		ErrorCode ReadFromStruct(const BYTE* i_lpBuffer)
		{
			unsigned int bufferPos = 0;
			
			// 各アイテム
			for(unsigned int itemNum=0; itemNum<this->lpLayerConfigItem.size(); itemNum++)
			{
				IItemBase* pItemBase = this->GetItemByNum(itemNum);
				if(pItemBase == NULL)
					continue;

				bufferPos += pItemBase->ReadFromStruct(&i_lpBuffer[bufferPos]);
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** 空のレイヤー設定情報を作成する */
	extern "C" GRAVISBELL_SETTINGDATA_API IDataEx* CreateEmptyData(const GUID& layerCode, const VersionCode& versionCode)
	{
		return new Data(layerCode, versionCode);
	}

}	// Standard
}	// SettingData
}	// Gravisbell