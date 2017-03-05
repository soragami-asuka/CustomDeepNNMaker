//=======================================
// �o�[�W�����R�[�h
//=======================================
#ifndef __GRAVISBELL_VERSION_CODE_H__
#define __GRAVISBELL_VERSION_CODE_H__

#include"Common.h"

namespace Gravisbell {
	
	/** �o�[�W�����R�[�h */
	struct VersionCode
	{
		union
		{
			struct
			{
				U16 major;		/// ���W���[�o�[�W����	���i�����{����ύX����ꍇ�ɕύX����܂��B
				U16 minor;		/// �}�C�i�[�o�[�W����	�啝�Ȏd�l�ύX�E�@�\�ǉ�������ꍇ�ɕύX����܂��B
				U16 revision;	/// ���r�W����			�d�l�ύX�E�@�\�ǉ�������ꍇ�ɕύX����܂��B
				U16 build;		/// �r���h				�C���p�b�`���ƂɕύX����܂��B
			};
			U16 lpData[4];
		};

		/** �R���X�g���N�^ */
		VersionCode()
			:	VersionCode(0, 0, 0, 0)
		{
		}
		/** �R���X�g���N�^ */
		VersionCode(U16 major, U16 minor, U16 revision, U16 build)
			:	major	(major)
			,	minor	(minor)
			,	revision(revision)
			,	build	(build)
		{
		}
		/** �R�s�[�R���X�g���N�^ */
		VersionCode(const VersionCode& code)
			:	VersionCode(code.major, code.minor, code.revision, code.build)
		{
		}

		/** �����Z */
		const VersionCode& operator=(const VersionCode& code)
		{
			this->major = code.major;
			this->minor = code.minor;
			this->revision = code.revision;
			this->build = code.build;

			return *this;
		}

		/** ��v���Z */
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
		/** �s��v���Z */
		bool operator!=(const VersionCode& code)const
		{
			return !(*this == code);
		}
		/** ��r���Z */
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