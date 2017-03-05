//=======================================
// バージョンコード
//=======================================
#ifndef __GRAVISBELL_VERSION_CODE_H__
#define __GRAVISBELL_VERSION_CODE_H__

#include"Common.h"

namespace Gravisbell {
	
	/** バージョンコード */
	struct VersionCode
	{
		union
		{
			struct
			{
				U16 major;		/// メジャーバージョン	製品を根本から変更する場合に変更されます。
				U16 minor;		/// マイナーバージョン	大幅な仕様変更・機能追加をする場合に変更されます。
				U16 revision;	/// リビジョン			仕様変更・機能追加をする場合に変更されます。
				U16 build;		/// ビルド				修正パッチごとに変更されます。
			};
			U16 lpData[4];
		};

		/** コンストラクタ */
		VersionCode()
			:	VersionCode(0, 0, 0, 0)
		{
		}
		/** コンストラクタ */
		VersionCode(U16 major, U16 minor, U16 revision, U16 build)
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

}	// Gravisbell

#endif