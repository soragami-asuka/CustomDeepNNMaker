//==================================
// 設定項目(列挙型)
//==================================
#include "stdafx.h"

#include"NNLayerConfig.h"
#include"INNLayerConfigItemEx_Enum.h"
#include"NNLayerConfigItemBase.h"

#include<string>
#include<vector>

namespace CustomDeepNNLibrary
{
	class NNLayerConfigItem_Enum : public NNLayerConfigItemBase<INNLayerConfigItemEx_Enum>
	{
	private:
		struct EnumItem
		{
			std::string id;
			std::string name;
			std::string comment;

			/** コンストラクタ */
			EnumItem()
				: id ("")
				, name ("")
				, comment ("")
			{
			}
			/** コンストラクタ */
			EnumItem(const std::string& id, const std::string& name, const std::string& comment)
				: id		(id)
				, name		(name)
				, comment	(comment)
			{
			}
			/** コピーコンストラクタ */
			EnumItem(const EnumItem& item)
				: id		(item.id)
				, name		(item.name)
				, comment	(item.comment)
			{
			}

			/** =演算 */
			const EnumItem& operator=(const EnumItem& item)
			{
				this->id = item.id;
				this->name = item.name;
				this->comment = item.comment;

				return *this;
			}


			bool operator==(const EnumItem& item)const
			{
				if(this->id != item.id)
					return false;
				if(this->name != item.name)
					return false;
				if(this->name != item.comment)
					return false;
				return true;
			}
			bool operator!=(const EnumItem& item)const
			{
				return !(*this == item);
			}
		};

	private:
		std::vector<EnumItem> lpEnumItem;

		std::string value;
		std::string defaultValue;


	public:
		/** コンストラクタ */
		NNLayerConfigItem_Enum(const char i_szID[], const char i_szName[], const char i_szText[])
			: NNLayerConfigItemBase(i_szID, i_szName, i_szText)
		{
		}
		/** コピーコンストラクタ */
		NNLayerConfigItem_Enum(const NNLayerConfigItem_Enum& item)
			: NNLayerConfigItemBase(item)
			, lpEnumItem (item.lpEnumItem)
			, value (item.value)
			, defaultValue (item.defaultValue)
		{
		}
		/** デストラクタ */
		virtual ~NNLayerConfigItem_Enum(){}
		
		/** 一致演算 */
		bool operator==(const INNLayerConfigItemBase& item)const
		{
			// 種別の確認
			if(this->GetItemType() != item.GetItemType())
				return false;

			// アイテムを変換
			const NNLayerConfigItem_Enum* pItem = dynamic_cast<const NNLayerConfigItem_Enum*>(&item);
			if(pItem == NULL)
				return false;

			if(NNLayerConfigItemBase::operator!=(*pItem))
				return false;

			for(unsigned int itemNum=0; itemNum<this->lpEnumItem.size(); itemNum++)
			{
				if(this->lpEnumItem[itemNum] != pItem->lpEnumItem[itemNum])
					return false;
			}
			if(this->defaultValue != pItem->defaultValue)
				return false;

			if(this->value != pItem->value)
				return false;

			return true;
		}
		/** 不一致演算 */
		bool operator!=(const INNLayerConfigItemBase& item)const
		{
			return !(*this == item);
		}

		/** 自身の複製を作成する */
		INNLayerConfigItemBase* Clone()const
		{
			return new NNLayerConfigItem_Enum(*this);
		}

	public:
		/** 設定項目種別を取得する */
		NNLayerConfigItemType GetItemType()const
		{
			return CONFIGITEM_TYPE_ENUM;
		}

	public:
		/** 値を取得する */
		int GetValue()const
		{
			return this->GetNumByID(this->value.c_str());
		}

		/** 値を設定する
			@param value	設定する値
			@return 成功した場合0 */
		ELayerErrorCode SetValue(int value)
		{
			if(value < 0)
				return LAYER_ERROR_COMMON_OUT_OF_VALUERANGE;
			if(value >= (int)this->lpEnumItem.size())
				return LAYER_ERROR_COMMON_OUT_OF_VALUERANGE;

			this->value = this->lpEnumItem[value].id;

			return LAYER_ERROR_NONE;
		}
		/** 値を設定する(文字列指定)
			@param i_szID	設定する値(文字列指定)
			@return 成功した場合0 */
		ELayerErrorCode SetValue(const char i_szEnumID[])
		{
			return this->SetValue(this->GetNumByID(i_szEnumID));
		}

	public:
		/** 列挙要素数を取得する */
		unsigned int GetEnumCount()const
		{
			return this->lpEnumItem.size();
		}
		/** 列挙要素IDを番号指定で取得する.
			@param num		要素番号
			@param o_szBuf	格納先バッファ
			@return 成功した場合0 */
		int GetEnumID(int num, char o_szBuf[])const
		{
			if(num < 0)
				return -1;
			if(num >= (int)this->lpEnumItem.size())
				return -1;
			
			memcpy(o_szBuf, this->lpEnumItem[num].id.c_str(), this->lpEnumItem[num].id.size() + 1);

			return 0;
		}
		/** 列挙要素IDを番号指定で取得する.
			@param num		要素番号
			@param o_szBuf	格納先バッファ
			@return 成功した場合0 */
		int GetEnumName(int num, char o_szBuf[])const
		{
			if(num < 0)
				return -1;
			if(num >= (int)this->lpEnumItem.size())
				return -1;
			
			memcpy(o_szBuf, this->lpEnumItem[num].name.c_str(), this->lpEnumItem[num].name.size() + 1);

			return 0;
		}
		/** 列挙要素コメントを番号指定で取得する.
			@param num		要素番号
			@param o_szBuf	格納先バッファ
			@return 成功した場合0 */
		int GetEnumComment(int num, char o_szBuf[])const
		{
			if(num < 0)
				return -1;
			if(num >= (int)this->lpEnumItem.size())
				return -1;
			
			memcpy(o_szBuf, this->lpEnumItem[num].comment.c_str(), this->lpEnumItem[num].comment.size() + 1);

			return 0;
		}

		/** IDを指定して列挙要素番号を取得する
			@param i_szBuf　調べる列挙ID.
			@return 成功した場合0<=num<GetEnumCountの値. 失敗した場合は負の値が返る. */
		int GetNumByID(const char i_szEnumID[])const
		{
			std::string enumID = i_szEnumID;

			for(unsigned int itemNum=0; itemNum<this->lpEnumItem.size(); itemNum++)
			{
				if(this->lpEnumItem[itemNum].id == enumID)
					return (int)itemNum;
			}

			return -1;
		}

		/** デフォルトの設定値を取得する */
		int GetDefault()const
		{
			return this->GetNumByID(this->defaultValue.c_str());
		}

	public:
		/** 列挙値を追加する.
			同一IDが既に存在する場合失敗する.
			@param szEnumID		追加するID
			@param szEnumName	追加する名前
			@param szComment	追加するコメント分.
			@return 成功した場合、追加されたアイテムの番号が返る. 失敗した場合は負の値が返る. */
		int AddEnumItem(const char szEnumID[], const char szEnumName[], const char szComment[])
		{
			// 同一IDが存在するか確認
			int sameID = this->GetNumByID(szEnumID);
			if(sameID >= 0)
				return -1;

			std::string id = szEnumID;
			if(id.size()+1 >= ID_BUFFER_MAX)
				return -1;

			// 追加
			this->lpEnumItem.push_back(EnumItem(id, szEnumName, szComment));

			return this->lpEnumItem.size()-1;
		}

		/** 列挙値を削除する.
			@param num	削除する列挙番号
			@return 成功した場合0 */
		int EraseEnumItem(int num)
		{
			if(num < 0)
				return -1;
			if(num >= (int)this->lpEnumItem.size())
				return -1;

			// iteratorを進める
			auto it = this->lpEnumItem.begin();
			for(int i=0; i<num; i++)
				it++;

			// 削除
			this->lpEnumItem.erase(it);

			return 0;
		}
		/** 列挙値を削除する
			@param szEnumID 削除する列挙ID
			@return 成功した場合0 */
		int EraseEnumItem(const char szEnumID[])
		{
			return this->EraseEnumItem(this->GetNumByID(szEnumID));
		}

		/** デフォルト値を設定する.	番号指定.
			@param num デフォルト値に設定する番号. 
			@return 成功した場合0 */
		int SetDefaultItem(int num)
		{
			char szEnumID[ID_BUFFER_MAX];

			// IDを取得する
			if(this->GetEnumID(num, szEnumID) < 0)
				return -1;

			this->defaultValue = szEnumID;

			return 0;
		}
		/** デフォルト値を設定する.	ID指定. 
			@param szID デフォルト値に設定するID. 
			@return 成功した場合0 */
		int SetDefaultItem(const char szEnumID[])
		{
			if(this->GetNumByID(szEnumID) < 0)
				return -1;

			this->defaultValue = szEnumID;

			return 0;
		}


	public:
		/** 保存に必要なバイト数を取得する */
		unsigned int GetUseBufferByteCount()const
		{
			unsigned int byteCount = 0;

			byteCount += sizeof(unsigned int);		// 値のバッファサイズ
			byteCount += this->value.size();		// 値

			return byteCount;
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

			// 値
			{
				// バッファサイズ
				unsigned int bufferSize = *(unsigned int*)&i_lpBuffer[bufferPos];
				bufferPos += sizeof(unsigned int);

				// 値
				std::vector<char> szBuf(bufferSize+1, NULL);
				for(unsigned int i=0; i<bufferSize; i++)
				{
					szBuf[i] = i_lpBuffer[bufferPos++];
				}
				std::string value = &szBuf[0];


				this->SetValue(value.c_str());
			}


			return bufferPos;
		}
		/** バッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合0 */
		int WriteToBuffer(BYTE* o_lpBuffer)const
		{
			unsigned int bufferPos = 0;
			
			// 値
			{
				// バッファサイズ
				unsigned int bufferSize = this->value.size();;
				*(unsigned int*)&o_lpBuffer[bufferPos] = bufferSize;
				bufferPos += sizeof(unsigned int);

				// 値
				memcpy(&o_lpBuffer[bufferPos], this->value.c_str(), bufferSize);
				bufferPos += bufferSize;
			}


			return bufferPos;
		}
	};
	
	/** 設定項目(列挙値)を作成する */
	extern "C" NNLAYERCONFIG_API INNLayerConfigItemEx_Enum* CreateLayerCofigItem_Enum(const char i_szID[], const char i_szName[], const char i_szText[])
	{
		return new NNLayerConfigItem_Enum(i_szID, i_szName, i_szText);
	}
}