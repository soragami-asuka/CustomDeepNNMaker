// NNLayerConfig.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"
#include "NNLayerConfig.h"

#include<string>
#include<vector>

namespace CustomDeepNNLibrary
{
	class NNLayerConfig : public INNLayerConfigEx
	{
	private:
		GUID layerCode;
		VersionCode versionCode;

		std::vector<INNLayerConfigItemBase*> lpLayerConfigItem;

	public:
		/** コンストラクタ */
		NNLayerConfig(const GUID& layerCode, const VersionCode& versionCode)
			: INNLayerConfigEx()
			, layerCode		(layerCode)
			, versionCode	(versionCode)
		{
		}
		/** コピーコンストラクタ */
		NNLayerConfig(const NNLayerConfig& config)
			: layerCode	(config.layerCode)
			, versionCode (config.versionCode)
		{
			for(unsigned int itemNum=0; itemNum<config.lpLayerConfigItem.size(); itemNum++)
			{
				this->lpLayerConfigItem.push_back(config.lpLayerConfigItem[itemNum]->Clone());
			}
		}
		/** デストラクタ */
		virtual ~NNLayerConfig()
		{
			for(unsigned int itemNum=0; itemNum<lpLayerConfigItem.size(); itemNum++)
			{
				if(lpLayerConfigItem[itemNum] != NULL)
					delete lpLayerConfigItem[itemNum];
			}
		}

		/** 一致演算 */
		bool operator==(const INNLayerConfig& config)const
		{
			NNLayerConfig* pConfig = (NNLayerConfig*)&config;

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
				INNLayerConfigItemBase* pItemL = this->lpLayerConfigItem[itemNum];
				INNLayerConfigItemBase* pItemR = pConfig->lpLayerConfigItem[itemNum];

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
		bool operator!=(const INNLayerConfig& config)const
		{
			return !(*this == config);
		}

		/** 自身の複製を作成する */
		virtual INNLayerConfig* Clone()const
		{
			return new NNLayerConfig(*this);
		}

	public:
		/** レイヤー識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		ELayerErrorCode GetLayerCode(GUID& o_guid)const
		{
			o_guid = this->layerCode;

			return LAYER_ERROR_NONE;
		}

	public:
		/** 設定項目数を取得する */
		unsigned int GetItemCount()const
		{
			return this->lpLayerConfigItem.size();
		}
		/** 設定項目を番号指定で取得する */
		const INNLayerConfigItemBase* GetItemByNum(unsigned int num)const
		{
			if(num >= this->lpLayerConfigItem.size())
				return NULL;
			return this->lpLayerConfigItem[num];
		}
		
		/** アイテムを追加する.
			追加されたアイテムは内部でdeleteされる. */
		int AddItem(INNLayerConfigItemBase* pItem)
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
				bufferSize += sizeof(NNLayerConfigItemType);

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



			std::vector<INNLayerConfigItemBase*> lpTmpItem;
			for(unsigned int itemNum=0; itemNum<itemCount; itemNum++)
			{
				// アイテム種別
				NNLayerConfigItemType itemType = *(NNLayerConfigItemType*)&i_lpBuffer[bufferPos];
				bufferPos += sizeof(NNLayerConfigItemType);

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
				*(NNLayerConfigItemType*)&o_lpBuffer[bufferPos] = this->lpLayerConfigItem[itemNum]->GetItemType();
				bufferPos += sizeof(NNLayerConfigItemType);

				// アイテム
				bufferPos += this->lpLayerConfigItem[itemNum]->WriteToBuffer(&o_lpBuffer[bufferPos]);
			}

			return bufferPos;
		}
	};

	/** 空のレイヤー設定情報を作成する */
	extern "C" NNLAYERCONFIG_API INNLayerConfigEx* CreateEmptyLayerConfig(const GUID& layerCode, const VersionCode& versionCode)
	{
		return new NNLayerConfig(layerCode, versionCode);
	}
}