//===========================
// NNのレイヤー設定項目フォーマットベース
//===========================
#ifndef __I_NN_LAYER_CONFIG_H__
#define __I_NN_LAYER_CONFIG_H__

#include<guiddef.h>

#include"LayerErrorCode.h"

#ifndef BYTE
typedef unsigned char BYTE;
#endif

namespace CustomDeepNNLibrary
{
	/** 設定項目種別 */
	enum NNLayerConfigItemType
	{
		CONFIGITEM_TYPE_FLOAT,
		CONFIGITEM_TYPE_INT,
		CONFIGITEM_TYPE_STRING,
		CONFIGITEM_TYPE_BOOL,
		CONFIGITEM_TYPE_ENUM,

		CONFIGITEM_TYPE_COUNT
	};

	static const unsigned int CONFIGITEM_NAME_MAX = 64;
	static const unsigned int LAYERTYPE_CODE_MAX  = 36 + 1;	/**< レイヤーの種類を識別するためのコードの最大文字数(36文字＋nullコード) */

	/** バージョンコード */
	struct VersionCode
	{
		unsigned short major;		/// メジャーバージョン	製品を根本から変更する場合に変更されます。
		unsigned short minor;		/// マイナーバージョン	大幅な仕様変更・機能追加をする場合に変更されます。
		unsigned short revision;	/// リビジョン			仕様変更・機能追加をする場合に変更されます。
		unsigned short build;		/// ビルド				修正パッチごとに変更されます。

		/** コンストラクタ */
		VersionCode()
			:	VersionCode(0, 0, 0, 0)
		{
		}
		/** コンストラクタ */
		VersionCode(unsigned short major, unsigned short minor, unsigned short revision, unsigned short build)
			:	major	(major)
			,	minor	(minor)
			,	revision(revision)
			,	build	(build)
		{
		}
		/** コピーコンストラクタ */
		VersionCode(const VersionCode& code)
			:	VersionCode(code.major, code.minor, code.revision, code.build)
		{
		}

		/** ＝演算 */
		const VersionCode& operator=(const VersionCode& code)
		{
			this->major = code.major;
			this->minor = code.minor;
			this->revision = code.revision;
			this->build = code.build;

			return *this;
		}

		/** 一致演算 */
		bool operator==(const VersionCode& code)const
		{
			if(this->major != code.major)
				return false;
			if(this->minor != code.minor)
				return false;
			if(this->revision != code.revision)
				return false;
			if(this->build != code.build)
				return false;

			return true;
		}
		/** 不一致演算 */
		bool operator!=(const VersionCode& code)const
		{
			return !(*this == code);
		}
		/** 比較演算 */
		bool operator<(const VersionCode& code)const
		{
			if(this->major < code.major)
				return true;
			if(this->major > code.major)
				return false;

			if(this->minor < code.minor)
				return true;
			if(this->minor > code.minor)
				return false;

			if(this->revision < code.revision)
				return true;
			if(this->revision > code.revision)
				return false;

			if(this->build < code.build)
				return true;
			if(this->build > code.build)
				return false;

			return false;
		}
	};


	/** 設定項目のベース */
	class INNLayerConfigItemBase
	{
	public:
		/** コンストラクタ */
		INNLayerConfigItemBase(){};
		/** デストラクタ */
		virtual ~INNLayerConfigItemBase(){};

		/** 一致演算 */
		virtual bool operator==(const INNLayerConfigItemBase& item)const = 0;
		/** 不一致演算 */
		virtual bool operator!=(const INNLayerConfigItemBase& item)const = 0;

		/** 自身の複製を作成する */
		virtual INNLayerConfigItemBase* Clone()const = 0;

	public:
		/** 設定項目種別を取得する */
		virtual NNLayerConfigItemType GetItemType()const = 0;
		/** 項目名を取得する.
			@param o_szNameBuf	名前を格納するバッファ. CONFIGITEM_NAME_MAXのバイト数が必要 */
		virtual ELayerErrorCode GetConfigName(char o_szNameBuf[])const = 0;
		/** 項目IDを取得する.
			IDはLayerConfig内において必ずユニークである.
			@param o_szIDBuf	IDを格納するバッファ.CONFIGITEM_NAME_MAXのバイト数が必要　*/
		virtual ELayerErrorCode GetConfigID(char o_szIDBuf[])const = 0;

	public:
		/** 保存に必要なバイト数を取得する */
		virtual unsigned int GetUseBufferByteCount()const = 0;

		/** バッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合0 */
		virtual int WriteToBuffer(BYTE* o_lpBuffer)const = 0;
		/** バッファから読み込む.
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
		virtual int ReadFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize) = 0;
	};

	/** 設定項目(実数) */
	class INNLayerConfigItem_Float : public INNLayerConfigItemBase
	{
	public:
		/** コンストラクタ */
		INNLayerConfigItem_Float()
			: INNLayerConfigItemBase()
		{};
		/** デストラクタ */
		virtual ~INNLayerConfigItem_Float(){};

	public:
		/** 値を取得する */
		virtual float GetValue()const = 0;
		/** 値を設定する
			@param value	設定する値
			@return 成功した場合0 */
		virtual ELayerErrorCode SetValue(float value) = 0;

	public:
		/** 設定可能最小値を取得する */
		virtual float GetMin()const = 0;
		/** 設定可能最大値を取得する */
		virtual float GetMax()const = 0;

		/** デフォルトの設定値を取得する */
		virtual float GetDefault()const = 0;
	};

	/** 設定項目(整数) */
	class INNLayerConfigItem_Int : public INNLayerConfigItemBase
	{
	public:
		/** コンストラクタ */
		INNLayerConfigItem_Int()
			: INNLayerConfigItemBase()
		{};
		/** デストラクタ */
		virtual ~INNLayerConfigItem_Int(){};

	public:
		/** 値を取得する */
		virtual int GetValue()const = 0;
		/** 値を設定する
			@param value	設定する値
			@return 成功した場合0 */
		virtual ELayerErrorCode SetValue(int value) = 0;
		
	public:
		/** 設定可能最小値を取得する */
		virtual int GetMin()const = 0;
		/** 設定可能最大値を取得する */
		virtual int GetMax()const = 0;

		/** デフォルトの設定値を取得する */
		virtual int GetDefault()const = 0;
	};

	/** 設定項目(文字列) */
	class INNLayerConfigItem_String : public INNLayerConfigItemBase
	{
	public:
		/** コンストラクタ */
		INNLayerConfigItem_String()
			: INNLayerConfigItemBase()
		{};
		/** デストラクタ */
		virtual ~INNLayerConfigItem_String(){};

	public:
		/** 文字列の長さを取得する */
		virtual unsigned int GetLength()const = 0;
		/** 値を取得する.
			@param o_szBuf	格納先バッファ
			@return 成功した場合0 */
		virtual int GetValue(char o_szBuf[])const = 0;
		/** 値を設定する
			@param i_szBuf	設定する値
			@return 成功した場合0 */
		virtual ELayerErrorCode SetValue(const char i_szBuf[]) = 0;
		
	public:
		/** デフォルトの設定値を取得する
			@param o_szBuf	格納先バッファ
			@return 成功した場合0 */
		virtual int GetDefault(char o_szBuf[])const = 0;
	};

	/** 設定項目(論理値) */
	class INNLayerConfigItem_Bool : public INNLayerConfigItemBase
	{
	public:
		/** コンストラクタ */
		INNLayerConfigItem_Bool()
			: INNLayerConfigItemBase()
		{};
		/** デストラクタ */
		virtual ~INNLayerConfigItem_Bool(){};

	public:
		/** 値を取得する */
		virtual bool GetValue()const = 0;
		/** 値を設定する
			@param value	設定する値
			@return 成功した場合0 */
		virtual ELayerErrorCode SetValue(bool value) = 0;

	public:
		/** デフォルトの設定値を取得する */
		virtual bool GetDefault()const = 0;
	};

	/** 設定項目(列挙型).
		※列挙要素ID=一意の英数字. 列挙要素名=利用者が認識しやすい名前.文字列.(重複可).1行. コメント=列挙要素の説明文.複数行可 */
	class INNLayerConfigItem_Enum : public INNLayerConfigItemBase
	{
	public:
		static const unsigned int ID_BUFFER_MAX = 64;			/**< 列挙要素IDの最大バイト数 */
		static const unsigned int NAME_BUFFER_MAX = 64;			/**< 列挙要素名の最大バイト数 */
		static const unsigned int COMMENT_BUFFER_MAX = 256;		/**< コメントの最大バイト数 */

	public:
		/** コンストラクタ */
		INNLayerConfigItem_Enum()
			: INNLayerConfigItemBase()
		{};
		/** デストラクタ */
		virtual ~INNLayerConfigItem_Enum(){};

	public:
		/** 値を取得する */
		virtual int GetValue()const = 0;

		/** 値を設定する
			@param value	設定する値 */
		virtual ELayerErrorCode SetValue(int value) = 0;
		/** 値を設定する(文字列指定)
			@param i_szID	設定する値(文字列指定) */
		virtual ELayerErrorCode SetValue(const char i_szEnumID[]) = 0;

	public:
		/** 列挙要素数を取得する */
		virtual unsigned int GetEnumCount()const = 0;
		/** 列挙要素IDを番号指定で取得する.
			@param num		要素番号
			@param o_szBuf	格納先バッファ
			@return 成功した場合0 */
		virtual int GetEnumID(int num, char o_szBuf[])const = 0;
		/** 列挙要素IDを番号指定で取得する.
			@param num		要素番号
			@param o_szBuf	格納先バッファ
			@return 成功した場合0 */
		virtual int GetEnumName(int num, char o_szBuf[])const = 0;
		/** 列挙要素コメントを番号指定で取得する.
			@param num		要素番号
			@param o_szBuf	格納先バッファ
			@return 成功した場合0 */
		virtual int GetEnumComment(int num, char o_szBuf[])const = 0;

		/** IDを指定して列挙要素番号を取得する
			@param i_szBuf　調べる列挙ID.
			@return 成功した場合0<=num<GetEnumCountの値. 失敗した場合は負の値が返る. */
		virtual int GetNumByID(const char i_szBuf[])const = 0;


		/** デフォルトの設定値を取得する */
		virtual int GetDefault()const = 0;
	};



	/** 設定情報 */
	class INNLayerConfig
	{
	public:
		/** コンストラクタ */
		INNLayerConfig(){}
		/** デストラクタ */
		virtual ~INNLayerConfig(){}

		/** 一致演算 */
		virtual bool operator==(const INNLayerConfig& config)const = 0;
		/** 不一致演算 */
		virtual bool operator!=(const INNLayerConfig& config)const = 0;

		/** 自身の複製を作成する */
		virtual INNLayerConfig* Clone()const = 0;

	public:
		/** レイヤー識別コードを取得する.
			@param o_guid	格納先バッファ
			@return 成功した場合0 */
		virtual ELayerErrorCode GetLayerCode(GUID& o_guid)const = 0;

	public:
		/** 設定項目数を取得する */
		virtual unsigned int GetItemCount()const = 0;
		/** 設定項目を番号指定で取得する */
		virtual const INNLayerConfigItemBase* GetItemByNum(unsigned int i_num)const = 0;
		/** 設定項目をID指定で取得する */
		virtual const INNLayerConfigItemBase* GetItemByID(const char i_szIDBuf[])const = 0;

	public:
		/** 保存に必要なバイト数を取得する */
		virtual unsigned int GetUseBufferByteCount()const = 0;

		/** バッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合0 */
		virtual int WriteToBuffer(BYTE* o_lpBuffer)const = 0;
	};


}	// namespace CustomDeepNNLibrary



#endif	// __I_NN_LAYER_CONFIG_H__